import numpy as np
import sys
import codecs, json
from matplotlib import pyplot as plt
import pandas as pd
import os
import warnings
import skvideo.io as skv
from skimage.color import rgb2gray
from skimage.transform import resize
from unprojection import reproject
from eyefitter import SingleEyeFitter
from utils import save_json, load_json, convert_vec2angle31
from visualisation import draw_circle, draw_ellipse, draw_line, VideoManager
from model.DeepVOG_model import load_DeepVOG

class gaze_inferer(object):

    # A Singleton class single instance
    __instance = None
    
    """ Virtually private constructor. """
    def __init__(self, model=None, flen=6, ori_video_shape=(240,320), sensor_size=(3.6,4.8), infer_gaze_flag=True):
        """
        Initialize necessary parameters and load deep_learning model
        
        Args:
            model: Deep learning model that perform image segmentation. Pre-trained model is provided at https://github.com/pydsgz/DeepVOG/model/DeepVOG_model.py, simply by loading load_DeepVOG() with "DeepVOG_weights.h5" in the same directory. If you use your own model, it should take input of grayscale image (m, 240, 320, 3) with value float [0,1] and output (m, 240, 320, 3) with value float [0,1] where (m, 240, 320, 1) is the pupil map.
            
            flen (float): Focal length of camera in mm. You can look it up at the product menu of your camera
            
            ori_video_shape (tuple or list or np.ndarray): Original video shape from your camera, (height, width) in pixel. If you cropped the video before, use the "original" shape but not the cropped shape
            
            sensor_size (tuple or list or np.ndarray): Sensor size of your camera, (height, width) in mm. For 1/3 inch CMOS sensor, it should be (3.6, 4.8). Further reference can be found in https://en.wikipedia.org/wiki/Image_sensor_format and you can look up in your camera product menu
        

        """
        
        if gaze_inferer.__instance != None:
            return gaze_inferer.get_instance()
            
        gaze_inferer.__instance = self
            
        # Assertion of shape
        try:
            assert ((isinstance(flen, int) or isinstance(flen, float)))
            assert (isinstance(ori_video_shape, tuple) or isinstance(ori_video_shape, list) or isinstance(ori_video_shape, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size, np.ndarray))
        except AssertionError:
            print("At least one of your arguments does not have correct type")
            raise TypeError
            
        # Parameters dealing with camera and video shape
        self.flen = flen
        self.ori_video_shape, self.sensor_size = np.array(ori_video_shape).squeeze(), np.array(sensor_size).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape) / np.linalg.norm(self.sensor_size)
    
        self.model = load_DeepVOG()
        self.confidence_fitting_threshold = 0.96
        self.eyefitter = SingleEyeFitter(focal_length=self.flen * self.mm2px_scaling,
                                        pupil_radius=2 * self.mm2px_scaling,
                                        initial_eye_z=50 * self.mm2px_scaling)
        self.infer_gaze_flag = infer_gaze_flag
            
            
    @staticmethod 
    def get_instance():
        """ Static access method. """
        
        if gaze_inferer.__instance == None:
            return gaze_inferer()
            
        return gaze_inferer.__instance
    
    def http_image_inference(self, image, result_path="./inference.json"):
            h, w, c = 0, 0, 0
            if len(image.shape) == 2:
                h, w = image.shape
            elif len(image.shape) == 3:
                h, w, c = image.shape
            else:
                print("Invalid image shape")
                sys.exit(1)

            image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))
            frame_preprocessed = self._preprocess_image(image, None)
            
            self.load_eyeball_model("/notebooks/eyeball_model.json")

            # (predict) Check if the eyeball model is imported
            self._check_eyeball_model_exists()

            # Correct eyefitter's parameters in accordance with the image resizing
            self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
            self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

            # Set batch-wise operation details
            X_batch = np.zeros((1, 240, 320, 3))
            X_batch[0, :, :, :] = frame_preprocessed
            
            Y_batch = self.model.predict(X_batch)

            if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
                raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")
                
            _, _, _, inference_dict = self.infer_image(X=X_batch, Y=Y_batch)
            center_dict = dict()
            # center_dict['image_inference'] = inference_dict['image_inference'].tolist()
            center_dict['pupil_center'] = str(inference_dict['pupil_center'][0]) + ", " + str(inference_dict['pupil_center'][1])
            
            return json.dumps(center_dict)
        
        
    def process(self, input_src, mode="Infer", output_record_path="", batch_size=32,
                output_path="", heatmap=False, print_prefix=""):
        """

        Parameters
        ----------
        input_src : str
            Path of the video from which you want to (1) fit the eyeball model or (2) infer the gaze.
        mode : str
            There are two modes: "Fit" or "Infer". "Fit" will fit an eyeball model from the video source.
            "Infer" will infer the gaze from the video source.
        batch_size : int
            Batch size. Recommended >= 32.
        output_record_path : str
            Path of the csv file of your gaze estimation result. Only matter if the mode == "Infer".
        output_path : str
            Path of the output visualization video. If mode == "Fit", it draws segmented pupil ellipse.
            If mode == "Infer", it draws segmented pupil ellipse and gaze vector. if output_path == "",
            no visualization will be produced.
        heatmap : bool
            If True, show heatmap in the visualization video. If False, no heatmap will be shown.
        print_prefix : str
            What to print before the progress text.

        Returns
        -------
        None

        """
        
        """
        ------------------ UPDATES ------------------
        This is a different version of DeepVOG model. The parameters described above are the same parameters for the DeepVOG fitting mode (fitting is made on an input video). As for the 'Infer' mode - it was changed to an image inference instead of a video one: The input is a single frame and the output is the pupil's center coordinates of the given input frame. These coordinates are the only inference predicted by the model.
        """
        
        if mode == "Infer":
            # Infer one image each request
            batch_size = 1
            
            frame = plt.imread(input_src)
            if len(frame.shape) == 2:
                h, w = frame.shape
            elif len(frame.shape) == 3:
                h, w, c = frame.shape

            image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))
            frame_preprocessed = self._preprocess_image(frame, None)

            # (predict) Check if the eyeball model is imported
            self._check_eyeball_model_exists()

            # Correct eyefitter's parameters in accordance with the image resizing
            self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
            self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

            # Set batch-wise operation details
            X_batch = np.zeros((1, 240, 320, 3))
            X_batch[0, :, :, :] = frame_preprocessed
                
            Y_batch = self.model.predict(X_batch) # one frame --> in one batch --> in one video 

            _, _, _, r = self.infer_image(X=X_batch, Y=Y_batch, img_id=str(input_src[input_src.rfind('/') + 1 : input_src.rfind('.')]), csv_path=str(output_record_path), out_path=str(output_path))

            if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
                raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")

        #  ------------------- Fitting the model on a given video -------------------
        
        elif mode == "Fit":
            # Get video information (path strings, frame reader, video's shapes...etc)
            video_name_root, ext, vreader, vid_shapes, shape_correct, image_scaling_factor = self._get_video_info(input_src)
            (vid_m, vid_w, vid_h, vid_channels) = vid_shapes

            self.vid_manager = VideoManager(vreader=vreader, output_record_path=output_record_path,
                                            output_path=output_path, heatmap=heatmap)

            # Correct eyefitter's parameters in accordance with the image resizing
            self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
            self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

            # Set batch-wise operation details
            initial_frame, final_frame = 0, vid_m
            final_batch_size = vid_m % batch_size
            final_batch_idx = vid_m - final_batch_size
            X_batch = np.zeros((batch_size, 240, 320, 3))
            X_batch_final = np.zeros((vid_m % batch_size, 240, 320, 3))

            # Start looping for batch-wise processing
            for idx, frame in enumerate(vreader.nextFrame()):

                #print("\r%s%s %s (%d/%d)" % (print_prefix, mode, video_name_root + ext, idx + 1, vid_m), end="", flush=True)

                frame_preprocessed = self._preprocess_image(frame, shape_correct)
                mini_batch_idx = idx % batch_size

                # Before reaching the batch size, stack the array
                if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                    X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

                # After reaching the batch size, but not the final batch, predict heatmap and fit/infer angles
                elif (mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx):
                    Y_batch = self.model.predict(X_batch)
                    self._fitting_batch(X_batch=X_batch, Y_batch=Y_batch)

                    # Renew X_batch for next batch
                    X_batch = np.zeros((batch_size, 240, 320, 3))
                    X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

                # Within the final batch but not yet reaching the last index, stack the array
                elif (idx > final_batch_idx) and (idx != final_frame - 1):
                    X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed

                # Within the final batch and reaching the last index, predict heatmap and fit/infer angles
                elif idx == final_frame - 1:
                    X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
                    Y_batch = self.model.predict(X_batch_final)
                    self._fitting_batch(X_batch=X_batch_final, Y_batch=Y_batch)

            # Fit eyeball models. Parameters are stored as internal attributes of Eyefitter instance.
            _ = self.eyefitter.fit_projected_eye_centre(ransac=True, max_iters=100, min_distance=10*vid_m)
            _, _ = self.eyefitter.estimate_eye_sphere()

            # Issue error if eyeball model still does not exist after fitting.
            if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
                raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")
        
        print()

    def save_eyeball_model(self, path):
        """
        Save eyeball model parameters in json format.
        
        Args:
            path (str): path of the eyeball model file.
        """

        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            print("3D eyeball model not found. You may need -v or -m argument to check whether the pupil segmentation works properly.")
            raise
        else:
            save_dict = {"eye_centre": self.eyefitter.eye_centre.tolist(),
                         "aver_eye_radius": self.eyefitter.aver_eye_radius}
            save_json(path, save_dict)

    def load_eyeball_model(self, path):
        """
        Load eyeball model parameters of json format from path.
        
        Args:
            path (str): path of the eyeball model file.
        """
        loaded_dict = load_json(path)
        if (self.eyefitter.eye_centre is not None) or (self.eyefitter.aver_eye_radius is not None):
            warnings.warn("3D eyeball exists and reloaded")

        self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]

    def _fitting_batch(self, X_batch, Y_batch):
        if self.vid_manager.output_video_flag:
            # Convert video frames to 8 bit integer format for drawing the output video frames
            video_frames_batch = np.around(X_batch * 255).astype(int)
            vid_frame_shape_2d = (video_frames_batch.shape[1], video_frames_batch.shape[2])

        for batch_idx, (X_each, Y_each) in enumerate(zip(X_batch, Y_batch)):
            pred_each = Y_each[:, :, 1]
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred_each)
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info

            # If visualization is true, initialize output frame
            if self.vid_manager.output_video_flag:
                vid_frame = video_frames_batch[batch_idx,]

            # Fit each observation to eyeball model
            if centre is not None:
                if (ellipse_confidence > self.confidence_fitting_threshold):
                    self.eyefitter.add_to_fitting()

                # Draw ellipse and pupil centre on input video if visualization is enabled
                if self.vid_manager.output_video_flag:
                    ellipse_centre_np = np.array(centre)

#                     # Draw pupil ellipse
#                     vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, ellipse_info=ellipse_info, color=[255, 255, 0])
                    # Draw small circle at the ellipse centre
                    vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, centre=ellipse_centre_np, radius=5, color=[0, 255, 0])

                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)
            else:
                # Draw original input frame when no ellipse is found
                if self.vid_manager.output_video_flag:
                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)
                             
    def infer_image(self, X, Y, csv_path=None, out_path=None):
        csv_dict, result = dict(), dict()
        image = np.around(X * 255).astype(int)
        shape_2d = (image.shape[1], image.shape[2])
        Y = np.squeeze(Y)

        _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(Y[:, :, 1])
        (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
        
        image = image[0,]

        # If ellipse fitting is successful, i.e. an ellipse is located, AND gaze inference is ENABLED
        if (centre is not None) and self.infer_gaze_flag:
            p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
            p1, n1 = p_list[0], n_list[0]
            px, py, pz = p1[0, 0], p1[1, 0], p1[2, 0]
            x, y = convert_vec2angle31(n1)
            positions = (px, py, pz, centre[0], centre[1])  # Pupil 3D positions and 2D projected positions
            gaze_angles = (x, y)  # horizontal and vertical gaze angles
            inference_confidence = (ellipse_confidence, consistence)
            ellipse_centre_np = np.array(centre)
            #result['center'] = ellipse_centre_np

            # Code below is for drawing image
            projected_eye_centre = reproject(self.eyefitter.eye_centre,
                                            self.eyefitter.focal_length)  # shape (2,1)
                
            # The lines below are for translation from camera coordinate system (centred at image centre)
            # to numpy's indexing frame. You substract the vector by the half of the video's 2D shape.
            # Col = x-axis, Row = y-axis
            projected_eye_centre += np.array(shape_2d[::-1]).reshape(-1, 1) / 2
            image = draw_circle(output_frame=image, frame_shape=shape_2d,
                                centre=ellipse_centre_np, radius=5, color=[255, 0, 0])
            image = image.astype(np.uint8)
            if out_path is not None:
                plt.imsave(out_path, image)
                
            result['image_inference'] = image
            result['pupil_center'] = ellipse_centre_np
            
            if csv_path is not None:
                df = pd.DataFrame.from_dict(result)
                df.to_csv(str(csv_path)) 
            
        return positions, gaze_angles, inference_confidence, result

    def _get_video_info(self, input_src):
        video_name_with_ext = os.path.split(input_src)[1]
        video_name_root, ext = os.path.splitext(video_name_with_ext)
        vreader = skv.FFmpegReader(input_src)
        m, w, h, channels = vreader.getShape()
        image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))
        shape_correct = self._inspectVideoShape(w, h)
        return video_name_root, ext, vreader, (m, w, h, channels), shape_correct, image_scaling_factor

    def _check_eyeball_model_exists(self):
        try:
            if self.infer_gaze_flag:
                assert isinstance(self.eyefitter.eye_centre, np.ndarray)
                assert self.eyefitter.eye_centre.shape == (3, 1)
                assert self.eyefitter.aver_eye_radius is not None
            else:
                pass
        except AssertionError as e:
            print(
                "3D eyeball mode is not found. Gaze inference cannot continue. Please fit/load an eyeball model first")
            raise e

    @staticmethod
    def _inspectVideoShape(w, h):
        if (w, h) == (240, 320):
            return True
        else:
            return False

    @staticmethod
    def _computeCroppedShape(ori_video_shape, crop_size):
        video = np.zeros(ori_video_shape)
        cropped = video[crop_size[0]:crop_size[1], crop_size[2], crop_size[3]]
        return cropped.shape

    @staticmethod
    def _preprocess_image(img, shape_correct):
        """
        
        Args:
            img (numpy array): unprocessed image with shape (w, h, 3) and values int [0, 255]
        Returns:
            output_img (numpy array): processed grayscale image with shape ( 240, 320, 1) and values float [0,1]
        """
        output_img = np.zeros((240, 320, 1))
        img = img / 255
        img = rgb2gray(img)
        if not shape_correct:
            img = resize(img, (240, 320))
        output_img[:, :, :] = img.reshape(240, 320, 1)
        return output_img
