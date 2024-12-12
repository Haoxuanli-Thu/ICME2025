import torch
from torch import autocast
import numpy as np
from torch.cuda.amp import autocast as dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch.nn as nn
from nnunetv2.training.loss.compound_losses import (
    DC_and_CE_loss,
    DC_and_BCE_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UXNet import UXNET
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import HCMA
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.moganet import MogaNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.SwinUNETRv2 import SwinUNETR
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UNETR import UNETR
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.AttentionUnet import AttentionUnet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.MedNeXt import MedNeXt
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.unetrpp import UNETR_PP
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnFormer import nnFormer
class nnUNetTrainerHCMA(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        exp_name='',
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, exp_name,device
        )
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 200
        self.batch_size = 2
        self.initial_lr = 4e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False  # Truse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.initial_lr,
            epochs=self.num_epochs,
            pct_start=0.06,
            steps_per_epoch=self.num_iterations_per_epoch,
            anneal_strategy="linear",
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=8)}"
        )
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            fea,output = self.network(data)
            # del data
            l = self.loss(fea,output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            self.lr_scheduler.step()
        return {"loss": l.detach().cpu().numpy()}
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            fea,output = self.network(data)
            del data
            l = self.loss(fea, output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)
        """
        # model = Baselinev5_DenseDown(in_channels=1,n_classes=2,predict_mode=True)
        # model = Baselinev5(
        #     num_input_channels,
        #     num_output_channels,
        #     deep_supervision=enable_deep_supervision,
        # )
        # model = FrigeSelfAxialMamba(num_input_channels,2,predict_mode=False)
        # model = HCMA(num_input_channels,2,predict_mode=True)
        # model = AxialMamba(num_input_channels,2,predict_mode=False)
        # model = SingleBaselinev5(num_input_channels,2,predict_mode=True)
        # model = SingleMamba(num_input_channels,2,predict_mode=True)
        # model = DifferentPatch(num_input_channels,2,predict_mode=False)
        # model = nnFormer(input_channels=num_input_channels,num_classes=num_output_channels)
        # model = Baselinev5_DenseDown(
        #     num_input_channels,
        #     num_output_channels,
        #     deep_supervision=enable_deep_supervision,
        # )



     


        # model = UNETR_PP(
        #         in_channels=1,
        #         out_channels=2,  # 假设分割为2类
        #         feature_size=16,
        #         hidden_size=256,
        #         num_heads=8,
        #         pos_embed="perceptron",
        #         norm_name="instance",
        #         dropout_rate=0.1,
        #         depths=[3, 3, 3, 3],
        #         dims=[32, 64, 128, 256,512],
        #         conv_op=nn.Conv3d,
        #         do_ds=False,
        #         predict_mode=True
        #     )

        model=MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=2,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        predict_mode=True
        )


        
        # model = UNETR(num_input_channels, num_output_channels, img_size=(128, 128, 128))



        # model = SwinUNETR(
        #     img_size=(128, 128, 128),
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     use_v2=False,
        #     predict_mode=True
        # )



        # model = AttentionUnet(
        #     3,
        #     in_channels=num_input_channels,
        #     out_channels=num_output_channels,
        #     channels=[32, 64, 128, 256, 512],
        #     strides=[2, 2, 2, 2],
        #     predict_mode=True
        # )

        # model = UXNET(
        #     num_input_channels,
        #     num_output_channels,
        #     # deep_supervision=enable_deep_supervision,
        #     predict_mode=True
        # )
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
