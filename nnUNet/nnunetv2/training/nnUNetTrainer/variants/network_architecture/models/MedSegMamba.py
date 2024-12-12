# from fvcore.nn import FlopCountAnalysis
import ipdb
import torch
import torch.nn as nn
try:
    # from flopth import flopth
    from fvcore.nn import FlopCountAnalysis
except:
    pass
try:
    from VSS3D import VSSLayer3D
    from Transformer import TransformerBottleneck
except:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models.VSS3D import VSSLayer3D
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models.Transformer import TransformerBottleneck
class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class ResConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, id = True, preact = True): #id was False
        super().__init__()
        if preact:
            self.conv = nn.Sequential(
                nn.GroupNorm(8, ch_in),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                )
        id = (ch_in == ch_out) and id
        self.identity = (lambda x: x) if id else nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        x = self.conv(inp)
        residual = self.identity(inp)
        return residual + x
    
class upconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3, upsample = True, preact = True):
        super().__init__()
        if upsample:
            upconv = nn.Sequential(nn.Upsample(scale_factor = 2),
                                    nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            )
        else:
            upconv = nn.ConvTranspose3d(
                                        in_channels=ch_in,
                                        out_channels=ch_out,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=kernel_size//2 - (1 - kernel_size % 2),
                                        output_padding=kernel_size % 2,
                                        )
        if preact:
            act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True))# if act else (lambda x: x)
            self.up = nn.Sequential(act, upconv)
        else:
            act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True))# if act else (lambda x: x)
            self.up = nn.Sequential(upconv, act)
            
            
    def forward(self,x):
        x = self.up(x)
        return x
    
class downconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3, maxpool = False, act = True, preact = True):
        super().__init__()
        if maxpool:
            downconv = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                  nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True))
        else:
            downconv = nn.Conv3d(
                                in_channels=ch_in,
                                out_channels=ch_out,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size//2 - (1 - kernel_size % 2),
                                )
        if preact:
            act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True)) if act else (lambda x: x)
            self.down = nn.Sequential(act, downconv)
        else:
            act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True)) if act else (lambda x: x)
            self.down = nn.Sequential(downconv, act)
    def forward(self,x):
        x = self.down(x)
        return x

class down_block(nn.Module):
    def __init__(self, ch_in, ch_out, maxpool = False, id=True, preact = True, kernel_size = 3):
        super().__init__()
        if maxpool:
            downsample = nn.MaxPool3d(kernel_size=2,stride=2)
            resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        else:
            downconv = nn.Conv3d(
                                in_channels=ch_in,
                                out_channels=ch_out,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size//2 - (1 - kernel_size % 2),
                                )
            if preact:
                act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_out, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(act, downconv, resblock)
            else:
                act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(downconv, act, resblock)

    def forward(self,x):
        x = self.down(x)
        return x
    
class MedSegMamba(nn.Module):
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8, # 2^(number of downsample/upsample stages in the encoder/decoder)
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256,1024], #last element is the size which the bottleneck processes
        mamba_d_state = 64, # for vss block
        num_layers = 9,
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        drop_path_rate=0.3,
        ssm_expansion_factor=1,
        scan_type = 'scan',
        id = False, #whether to have conv1x1 in the residual path when input and output channel sizes of residual block is the same
        preact = False, #whether to have preactivation residual blocks (norm>act>conv vs conv>norm>act)
        maxpool = True, #maxpool vs strided conv
        upsample = True, #upsample vs transposed conv
        full_final_block = True, # found to do better when the last resblock is large on TABSurfer, but a lot more memory
        ):
        super().__init__()
        self.embedding_dim = channel_sizes[5]

        self.hidden_dim = int((img_dim // patch_dim) ** 3)
        
        self.preconv = nn.Conv3d(img_ch, channel_sizes[1], kernel_size=3,stride=1,padding=1) if preact else (lambda x: x)
        if preact:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[1],ch_out=channel_sizes[1], id=id, preact=preact) # 96
        else:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[0],ch_out=channel_sizes[1], id=id, preact=preact) # 96
            
        self.downconv1 = down_block(channel_sizes[1], channel_sizes[2], maxpool, id, preact)

        self.downconv2 = down_block(channel_sizes[2], channel_sizes[3], maxpool, id, preact)

        self.downconv3 = down_block(channel_sizes[3], channel_sizes[4], maxpool, id, preact)

        self.gn = nn.GroupNorm(8, channel_sizes[4])
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv3d(
                channel_sizes[4],
                channel_sizes[5],
                kernel_size=3,
                stride=1,
                padding=1
                )
        if vss_version == 'TABSurfer': # for when debugging CNN parts
            print('TABSurfer bottleneck', flush=True)
            self.bottleneck = TransformerBottleneck((img_dim//patch_dim), channel_sizes[5], num_heads = 16, num_layers=num_layers, dropout_rate=mlp_dropout_rate, attn_dropout_rate=attn_dropout_rate)
        else:
            self.bottleneck = VSSLayer3D(dim = self.embedding_dim,
                                     depth = num_layers,
                                     drop_path = drop_path_rate,
                                     attn_drop = attn_dropout_rate,
                                     mlp_drop=mlp_dropout_rate,
                                     d_state = mamba_d_state,
                                     version=vss_version,
                                     expansion_factor=ssm_expansion_factor,
                                     scan_type=scan_type)
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        # self.compress = ResConvBlock3D(ch_in=channel_sizes[5], ch_out=channel_sizes[4], id=id, preact=preact)
        self.compress = conv_block_3D(ch_in=channel_sizes[5], ch_out=channel_sizes[4])

        self.Up4 = upconv(ch_in=channel_sizes[4],ch_out=channel_sizes[3], upsample=upsample, preact=preact)
        self.Up_conv4 = ResConvBlock3D(ch_in=channel_sizes[4], ch_out=channel_sizes[3], id=id, preact=preact)
        self.Up3 = upconv(ch_in=channel_sizes[3],ch_out=channel_sizes[2], upsample=upsample, preact=preact)
        self.Up_conv3 = ResConvBlock3D(ch_in=channel_sizes[3], ch_out=channel_sizes[2], id=id, preact=preact)
        self.Up2 = upconv(ch_in=channel_sizes[2],ch_out=channel_sizes[1], upsample=upsample, preact=preact)

        if full_final_block:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=channel_sizes[2], id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d(channel_sizes[2],output_ch,kernel_size=1,stride=1,padding=0)
        elif output_ch<=channel_sizes[1]:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=channel_sizes[1], id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d(channel_sizes[1],output_ch,kernel_size=1,stride=1,padding=0)
        else:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=(output_ch+8-output_ch%8), id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d((output_ch+8-output_ch%8),output_ch,kernel_size=1,stride=1,padding=0)

        self.act = nn.Softmax(dim=1)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        input = self.preconv(input)
        x1 = self.Conv1(input)

        x2 = self.downconv1(x1)

        x3 = self.downconv2(x2)

        x = self.downconv3(x3)

        #bottle neck
        x = self.gn(x)
        x = self.relu(x)
        x = self.expand(x) # increase number of channels to embedding dim
        x = x.permute(0,2,3,4,1) # B H W D C

        x = self.bottleneck(x)
        
        x = self.pre_head_ln(x)
        x = x.permute(0,4,1,2,3) # B C H W D
        x = self.compress(x) # 1 256 12 12 12

        #decoding
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)
        return d1

class MedSegMamba_v2(nn.Module): # add vss3d modules in skip connections
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8, # 2^(number of downsample/upsample stages in the encoder/decoder)
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256,1024], #last element is the size which the bottleneck processes
        mamba_d_state = 16, # for vss block
        # num_layers = 9,
        skip_layers = [0,3,3,6], # instead of skip connections 012 345 012 345 012
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        drop_path_rate=0.1,
        ssm_expansion_factor=1,
        scan_type = 'scan',
        id = False, 
        preact = False,
        maxpool = True,
        upsample = True,
        full_final_block = True, 
        ):
        super().__init__()
        orientations = [i%6 for i in range(sum(skip_layers))]

        self.embedding_dim = channel_sizes[5]
        self.skip_layers = skip_layers
        self.hidden_dim = int((img_dim // patch_dim) ** 3)
        
        self.preconv = nn.Conv3d(img_ch, channel_sizes[1], kernel_size=3,stride=1,padding=1) if preact else (lambda x: x)
        if preact:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[1],ch_out=channel_sizes[1], id=id, preact=preact) # 96
        else:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[0],ch_out=channel_sizes[1], id=id, preact=preact) # 96
            
        self.downconv1 = down_block(channel_sizes[1], channel_sizes[2], maxpool, id, preact)

        self.downconv2 = down_block(channel_sizes[2], channel_sizes[3], maxpool, id, preact)

        self.downconv3 = down_block(channel_sizes[3], channel_sizes[4], maxpool, id, preact)

        self.gn = nn.GroupNorm(8, channel_sizes[4])
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv3d(
                channel_sizes[4],
                channel_sizes[5],
                kernel_size=3,
                stride=1,
                padding=1
                )
        if skip_layers[0]>0:
            self.skip1 = VSSLayer3D(dim = channel_sizes[1],
                                        depth = skip_layers[0],
                                        drop_path = drop_path_rate,
                                        attn_drop = attn_dropout_rate,
                                        mlp_drop=mlp_dropout_rate,
                                        d_state = mamba_d_state,
                                        version=vss_version,
                                        expansion_factor=ssm_expansion_factor,
                                        scan_type=scan_type,
                                        size = 96,
                                        orientation_order=orientations[:skip_layers[0]])
        if skip_layers[1]>0:
            self.skip2 = VSSLayer3D(dim = channel_sizes[2],
                                        depth = skip_layers[1],
                                        drop_path = drop_path_rate,
                                        attn_drop = attn_dropout_rate,
                                        mlp_drop=mlp_dropout_rate,
                                        d_state = mamba_d_state,
                                        version=vss_version,
                                        expansion_factor=ssm_expansion_factor,
                                        scan_type=scan_type,
                                        size = 96//2,
                                        orientation_order=orientations[skip_layers[0]:skip_layers[0]+skip_layers[1]])
        if skip_layers[2]>0:
            self.skip3 = VSSLayer3D(dim = channel_sizes[3],
                                        depth = skip_layers[2],
                                        drop_path = drop_path_rate,
                                        attn_drop = attn_dropout_rate,
                                        mlp_drop=mlp_dropout_rate,
                                        d_state = mamba_d_state,
                                        version=vss_version,
                                        expansion_factor=ssm_expansion_factor,
                                        scan_type=scan_type,
                                        size = 96//4,
                                        orientation_order=orientations[skip_layers[0]+skip_layers[1]:skip_layers[0]+skip_layers[1]+skip_layers[2]])
        
        # print('TABSurfer bottleneck', flush=True)
        # self.bottleneck = TransformerBottleneck((img_dim//patch_dim), channel_sizes[5], num_heads = 16, num_layers=8, dropout_rate=mlp_dropout_rate, attn_dropout_rate=attn_dropout_rate)
        
        self.bottleneck = VSSLayer3D(dim = self.embedding_dim,
                                     depth = skip_layers[3],
                                     drop_path = drop_path_rate,
                                     attn_drop = attn_dropout_rate,
                                     mlp_drop=mlp_dropout_rate,
                                     d_state = mamba_d_state,
                                     version=vss_version,
                                     expansion_factor=ssm_expansion_factor,
                                     scan_type=scan_type,
                                     size = 96//8, 
                                     orientation_order=orientations[skip_layers[0]+skip_layers[1]+skip_layers[2]:])
        
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        # self.compress = ResConvBlock3D(ch_in=channel_sizes[5], ch_out=channel_sizes[4], id=id, preact=preact)
        self.compress = conv_block_3D(ch_in=channel_sizes[5], ch_out=channel_sizes[4])

        self.Up4 = upconv(ch_in=channel_sizes[4],ch_out=channel_sizes[3], upsample=upsample, preact=preact)
        self.Up_conv4 = ResConvBlock3D(ch_in=channel_sizes[4], ch_out=channel_sizes[3], id=id, preact=preact)
        self.Up3 = upconv(ch_in=channel_sizes[3],ch_out=channel_sizes[2], upsample=upsample, preact=preact)
        self.Up_conv3 = ResConvBlock3D(ch_in=channel_sizes[3], ch_out=channel_sizes[2], id=id, preact=preact)
        self.Up2 = upconv(ch_in=channel_sizes[2],ch_out=channel_sizes[1], upsample=upsample, preact=preact)

        if full_final_block:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=channel_sizes[2], id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d(channel_sizes[2],output_ch,kernel_size=1,stride=1,padding=0)
        elif output_ch<=channel_sizes[1]:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=channel_sizes[1], id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d(channel_sizes[1],output_ch,kernel_size=1,stride=1,padding=0)
        else:
            self.Up_conv2 = ResConvBlock3D(ch_in=channel_sizes[2], ch_out=(output_ch+8-output_ch%8), id=id, preact=preact)
            self.Conv_1x1 = nn.Conv3d((output_ch+8-output_ch%8),output_ch,kernel_size=1,stride=1,padding=0)

        self.act = nn.Softmax(dim=1)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch

    def forward(self,input):
        # encoding path
        input = self.preconv(input)
        x1 = self.Conv1(input)

        x2 = self.downconv1(x1)
        ipdb.set_trace()
        if self.skip_layers[0]>0:
            x1 = self.skip1(x1.permute(0,2,3,4,1)).permute(0,4,1,2,3) #before concatenating to the decoder
        
        x3 = self.downconv2(x2)
        if self.skip_layers[1]>0:
            x2 = self.skip2(x2.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        x = self.downconv3(x3)
        if self.skip_layers[2]>0:
            x3 = self.skip3(x3.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        #bottle neck
        x = self.gn(x)
        x = self.relu(x)
        x = self.expand(x) # increase number of channels to embedding dim
        x = x.permute(0,2,3,4,1) # B H W D C

        x = self.bottleneck(x)
        
        x = self.pre_head_ln(x)
        x = x.permute(0,4,1,2,3) # B C H W D
        x = self.compress(x) # 1 256 12 12 12

        #decoding
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)
        return d1
if __name__=="__main__":
    import torch
    import torch.nn as nn

    # 简化后的网络初始化和前向传播示例
    def initialize_and_run():
        # 初始化网络
        model = MedSegMamba_v2(
            img_dim=96,
            patch_dim=8,
            img_ch=1,
            output_ch=32,
            channel_sizes=[1,32,64,128,256,1024]
        )
        
        # 创建输入数据
        batch_size = 1
        input_channels = 1
        depth = 96
        height = 96
        width = 96
        x = torch.randn(batch_size, input_channels, depth, height, width)
        
        # 前向传播
        output = model(x)
        
        # 打印输入输出形状
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        return output

    # 运行示例
    output = initialize_and_run()