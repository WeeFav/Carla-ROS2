#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import scipy.special
import numpy as np
import PIL

from sensor_msgs.msg import Image   
from carla_client_msgs.msg import Lanes
from geometry_msgs.msg import Point32

from carla_client.lane_detection.model import UFLDNet


class LaneDet(Node):
    def __init__(self):
        super().__init__('lanedet')

        # Get parameters

        # Fixed parameters
        self.img_w = 1280
        self.img_h = 720
        self.griding_num = 100
        self.cls_num_per_lane = 56 # number of row anchors
        self.num_lanes = 4
        self.num_cls = 4
        self.backbone = '18'
        self.carla_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
        self.model_path = "/home/marvi/ros2_ws/src/carla_client/models/ep049.pth"
        self.use_classification = True
        

        self.model = UFLDNet(
            pretrained=False,
            backbone=self.backbone,
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, self.num_lanes),
            cat_dim=(self.num_lanes, self.num_cls),
            use_aux=False, # we dont need auxiliary segmentation in testing
            use_classification=self.use_classification
        )
        self.model.cuda()

        # load model weights
        state_dict = torch.load(self.model_path, map_location='cuda')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
                
        self.model.load_state_dict(compatible_state_dict, strict=False)
        self.model.eval()

        # transform input image
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        self.camera_rgb_sub = self.create_subscription(Image, '/carla/hero/rgb/image', self.camera_rgb_callback, 10)
        self.publisher = self.create_publisher(Lanes, '/lanes', 10)


    def reshape_image(self, image):
        array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB, (H, W, C)
        return array # (H, W, C)
    

    def tuple_list_to_points(self, lane):
        points = []
        for x, y in lane:
            p = Point32()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0   # or use real z if you have it
            points.append(p)
        return points


    def camera_rgb_callback(self, msg):
        self.image_rgb = self.reshape_image(msg)
        img = PIL.Image.fromarray(self.image_rgb, mode="RGB") 
        lanes_list = self.predict(img)

        lanes = Lanes()
        lanes.outer_left  = self.tuple_list_to_points(lanes_list[0])
        lanes.inner_left  = self.tuple_list_to_points(lanes_list[1])
        lanes.inner_right = self.tuple_list_to_points(lanes_list[2])
        lanes.outer_right = self.tuple_list_to_points(lanes_list[3])

        self.publisher.publish(lanes)


    def predict(self, img: Image):
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # convert (H, W, C) to [C, H, W], range [0,1]
        # img_tensor = TF.resize(img_tensor, (288, 800))  # resize using functional API
        # img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = self.img_transforms(img)
        img = img.unsqueeze(dim=0).cuda()

        with torch.no_grad():
            out = self.model(img) # (batch_size, num_gridding, num_cls_per_lane, num_of_lanes)

        detection = out['det']
        
        col_sample = np.linspace(0, 800 - 1, self.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = detection[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :] # flips rows
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # removes the last class, which is often reserved for no lane / background.
        idx = np.arange(self.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0) # expectation / avg idx
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.griding_num] = 0
        out_j = loc # (56, 4)

        lanes_list = [[] for _ in range(4)]
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        p = (int(out_j[k, i] * col_sample_w * self.img_w / 800) - 1, int(self.img_h * (self.carla_row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1)
                        lanes_list[i].append(p)     

        return lanes_list


def main():
    rclpy.init()
    node = LaneDet()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
