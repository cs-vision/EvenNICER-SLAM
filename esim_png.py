import numpy as np
import cv2
import os
import glob
import esim_py

# from plot_virtual_events.py
def viz_events(events, resolution=(680, 1200)):
    pos_events = events[events[:,-1]==1]
    neg_events = events[events[:,-1]==-1]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), pos_events[:,-1]**2)
    np.add.at(image_neg, (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), neg_events[:,-1]**2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution), 
            image_neg.reshape(resolution), 
            np.zeros(resolution, dtype="uint8") 
        ], -1
    ) # * 50

    return image_rgb    


# constructor
contrast_threshold_pos = 0.1 # contrast thesholds for positive 
contrast_threshold_neg = 0.1 # and negative events
refractory_period = 1e-4 # minimum waiting period (in sec) before a pixel can trigger a new event
log_eps = 1e-3 # epsilon that is used to numerical stability within the logarithm
use_log = 1 # wether or not to use log intensity
esim = esim_py.EventSimulator(
    contrast_threshold_pos, 
    contrast_threshold_neg, 
    refractory_period, 
    log_eps, 
    use_log,  
    )

# setter, useful within a training loop
esim.setParameters(contrast_threshold_pos, contrast_threshold_neg, refractory_period, log_eps, use_log)

# # generate events from a sequence of images
# events_from_images = esim.generateFromFolder(
#     path_to_image_folder, # absolute path to folder that stores images in numbered order
#     path_to_timestamps    # absolute path to timestamps file containing one timestamp (in secs) for each 
# )

# # generate events from a video
# events_from_video = esim.generateFromVideo(
#     path_to_video_file,   # absolute path to video storing images
#     path_to_timestamps    # absolute path to timestamps file
# )

# # generate events from list of images and timestamps
# events_list_of_images = esim.generateFromStampedImageSequence(
#     list_of_image_files,   # list of absolute paths to images
#     list_of_timestamps     # list of timestamps in ascending order
# )

# generate Replica events
places = [
        # 'office0'
        #   'office0_dense9995',
           'office1',
           'office2',
           'office3',
           'office4',
           'room0',
           'room1',
           'room2'
          ]
# input_dir = '/scratch_net/biwidl311/shichen/nice-slam/Datasets/Replica'
input_dir = '/scratch_net/biwidl215/myamaguchi/rpg_vid2e-master/data/Replica'
# output_folder = '/scratch-second/shichen/replica_gt_png'
# output_folder = '/scratch_net/biwidl311/shichen/nice-slam/Datasets/replica_gt_png'
# output_folder = '/scratch_net/biwidl311/shichen/nice-slam/Datasets/dense_replica_gt_png'
output_folder = '/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/Datasets/replica_gt_png'
# print(os.path.exists(output_dir))

for place in places:
    list_of_image_files = sorted(glob.glob(f'{input_dir}/{place}/results/frame*.jpg'))
    n_frames = len(list_of_image_files) # 2000 for Replica
    print('starting to process', place, 'with', n_frames, 'frames...\n')
    # fps = 24
    fps = 24 * 5 # dense replica
    interval = 1 / fps
    # list_of_timestamps = np.linspace(0, n_frames * interval, num=n_frames, endpoint=False)
    list_of_timestamps = np.linspace(0, interval, num=2, endpoint=True)
    # len_sequence = 100 # consider the memory issue

    output_dir = os.path.join(output_folder, place)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_frames-1):
        events_list_of_images = esim.generateFromStampedImageSequence(
            list_of_image_files[i : i+2],   # list of absolute paths to images
            # list_of_timestamps[i : i+2]     # list of timestamps in ascending order
            list_of_timestamps              # list of timestamps in ascending order
        ) # each event: [x, y, t, polarity]

        # print_x, print_y = 701, 167
        # is_pixel = ((events_list_of_images[:, 0] == print_x) * (events_list_of_images[:, 1] == print_y))
        # print(i, events_list_of_images[is_pixel])
        # target_time = 8.13316778e-04
        # is_time = (events_list_of_images[:, 2] <= target_time)
        # print(i, events_list_of_images[is_time])

        # # save events
        # npy_filename = place + '_' + 'frame' + str(i) + '_' + str(i+1) + '.npy'
        # with open(os.path.join(output_dir, npy_filename), 'wb') as f:
        #     np.save(f, events_list_of_images)
        # print('saved', npy_filename)

        # visualization
        img_events = viz_events(events_list_of_images)
        #img_events = np.clip(viz_events(events_list_of_images) * 50, 0, 255) # visualize
        # filename = place + '_' + 'frame' + str(i) + '_' + str(i+1) + '.jpg'
        # filename = place + '_' + 'frame' + str(i).zfill(4) + '_' + str(i+1).zfill(4) + '.png'
        filename = place + '_' + 'frame' + str(i).zfill(6) + '_' + str(i+1).zfill(6) + '.png'
        cv2.imwrite(os.path.join(output_dir, filename), img_events, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print('saved', filename)