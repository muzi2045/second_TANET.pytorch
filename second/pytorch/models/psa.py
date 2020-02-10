
import torch
from torch import nn
from second.pytorch.models.voxel_encoder import get_paddings_indicator
from torchplus.tools import change_default_args
from torchplus.nn import Empty, GroupNorm, Sequential
import numpy as np

from second.pytorch.models.rpn import register_rpn


#Our Coarse-to-Fine network
@register_rpn
class PSA(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='psa'):
           

        """
        :param use_norm:
        :param num_class:
        :param layer_nums:
        :param layer_strides:
        :param num_filters:
        :param upsample_strides:
        :param num_upsample_filters:
        :param num_input_filters:
        :param num_anchor_per_loc:
        :param encode_background_as_zeros:
        :param use_direction_classifier:
        :param use_groupnorm:
        :param num_groups:
        :param use_bev:
        :param box_code_size:
        :param name:
        """
        super(PSA, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc   ## 2
        self._use_direction_classifier = use_direction_classifier  # True
        # self._use_bev = use_bev   # False
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:   # True
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        # if use_bev:
        #     self.bev_extractor = Sequential(
        #         Conv2d(6, 32, 3, padding=1),
        #         BatchNorm2d(32),
        #         nn.ReLU(),
        #         # nn.MaxPool2d(2, 2),
        #         Conv2d(32, 64, 3, padding=1),
        #         BatchNorm2d(64),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #     )
        #     block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )

        upsample_strides = np.round(upsample_strides).astype(np.int64)
        print("upsample_strides:", upsample_strides)

        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        # print("in_channel:", num_filters[0])
        # print("out_channel:", num_upsample_filters[0])
        # print("kernel size:", upsample_strides[0])
        # print("stride:", upsample_strides[0])

        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)


        ###################  refine
        self.bottle_conv = nn.Conv2d(sum(num_upsample_filters), sum(num_upsample_filters)//3, 1)

        self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
        self.block1_dec4x = nn.MaxPool2d(kernel_size=4)   ### C=64

        self.block2_dec2x = nn.MaxPool2d(kernel_size=2)  ### C=128
        self.block2_inc2x = ConvTranspose2d(num_filters[1],num_filters[0]//2,upsample_strides[1],stride=upsample_strides[1])  ### C=32

        self.block3_inc2x = ConvTranspose2d(num_filters[2],num_filters[1]//2,upsample_strides[1],stride=upsample_strides[1])    #### C=64
        self.block3_inc4x = ConvTranspose2d(num_filters[2],num_filters[0]//2,upsample_strides[2],stride=upsample_strides[2])   #### C=32

        self.fusion_block1 = nn.Conv2d(num_filters[0]+num_filters[0]//2+num_filters[0]//2, num_filters[0], 1)
        self.fusion_block2 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[1]//2, num_filters[1], 1)
        self.fusion_block3 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[2], num_filters[2], 1)


        self.refine_up1 = Sequential(
            ConvTranspose2d(num_filters[0],num_upsample_filters[0], upsample_strides[0],stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.refine_up2 = Sequential(
            ConvTranspose2d(num_filters[1],num_upsample_filters[1],upsample_strides[1],stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.refine_up3 = Sequential(
            ConvTranspose2d(num_filters[2],num_upsample_filters[2],upsample_strides[2], stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

        #######
        # C_Bottle = cfg.PSA.C_Bottle
        # C = cfg.PSA.C_Reudce
        C_Bottle = 128
        C = 32

        self.RF1 = Sequential(  # 3*3
            Conv2d(C_Bottle*2, C, kernel_size=1, stride=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle*2, kernel_size=3, stride=1, padding=1, dilation=1),
            BatchNorm2d(C_Bottle*2),
            nn.ReLU(inplace=True),
        )

        self.RF2 = Sequential(  # 5*5
            Conv2d(C_Bottle, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle, kernel_size=3, stride=1, padding=1, dilation=1),
            BatchNorm2d(C_Bottle),
            nn.ReLU(inplace=True),
        )

        self.RF3 = Sequential(  # 7*7
            Conv2d(C_Bottle//2, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle//2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C_Bottle//2),
            nn.ReLU(inplace=True),
        )

        self.concat_conv1 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)  ## kernel_size=3
        self.concat_conv2 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)

        self.refine_cls = nn.Conv2d(sum(num_upsample_filters),num_cls, 1)
        self.refine_loc = nn.Conv2d(sum(num_upsample_filters),num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.refine_dir = nn.Conv2d(sum(num_upsample_filters),num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        print("psa input shape:", x.shape)
        x1 = self.block1(x)
        up1 = self.deconv1(x1)

        x2 = self.block2(x1)
        up2 = self.deconv2(x2)
        x3 = self.block3(x2)
        up3 = self.deconv3(x3)
        coarse_feat = torch.cat([up1, up2, up3], dim=1)

        print("coarse_feat shape:", coarse_feat.shape)
        box_preds = self.conv_box(coarse_feat)
        cls_preds = self.conv_cls(coarse_feat)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(coarse_feat)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds


        ###############Refine:
        blottle_conv = self.bottle_conv(coarse_feat)

        x1_dec2x = self.block1_dec2x(x1)
        x1_dec4x = self.block1_dec4x(x1)

        x2_dec2x = self.block2_dec2x(x2)
        x2_inc2x = self.block2_inc2x(x2)

        x3_inc2x = self.block3_inc2x(x3)
        x3_inc4x = self.block3_inc4x(x3)

        concat_block1 = torch.cat([x1,x2_inc2x,x3_inc4x], dim=1)
        fusion_block1 = self.fusion_block1(concat_block1)

        concat_block2 = torch.cat([x1_dec2x,x2,x3_inc2x], dim=1)
        fusion_block2 = self.fusion_block2(concat_block2)

        concat_block3 = torch.cat([x1_dec4x,x2_dec2x,x3], dim=1)
        fusion_block3 = self.fusion_block3(concat_block3)

        refine_up1 = self.RF3(fusion_block1)
        refine_up1 = self.refine_up1(refine_up1)
        refine_up2 = self.RF2(fusion_block2)
        refine_up2 = self.refine_up2(refine_up2)
        refine_up3 = self.RF1(fusion_block3)
        refine_up3 = self.refine_up3(refine_up3)


        branch1_sum_wise = refine_up1 + blottle_conv
        branch2_sum_wise = refine_up2 + blottle_conv
        branch3_sum_wise = refine_up3 + blottle_conv

        concat_conv1 = self.concat_conv1(branch1_sum_wise)
        concat_conv2 = self.concat_conv2(branch2_sum_wise)
        concat_conv3 = self.concat_conv3(branch3_sum_wise)

        PSA_output = torch.cat([concat_conv1,concat_conv2,concat_conv3], dim=1)

        refine_cls_preds = self.refine_cls(PSA_output)
        refine_loc_preds = self.refine_loc(PSA_output)

        refine_loc_preds = refine_loc_preds.permute(0, 2, 3, 1).contiguous()
        refine_cls_preds = refine_cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict["Refine_box_preds"] =  refine_loc_preds
        ret_dict["Refine_cls_preds"] =  refine_cls_preds

        if self._use_direction_classifier:
            refine_dir_preds = self.refine_dir(PSA_output)
            refine_dir_preds = refine_dir_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["Refine_dir_preds"] = refine_dir_preds

        return ret_dict