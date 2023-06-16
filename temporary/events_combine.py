import glob
import numpy as np
import time

event_folder = '/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/Datasets/replica_gt_events/room0'
# def main():
#     data_list = []
#     events_paths = sorted(
#         glob.glob(f'{event_folder}/*events*.txt')
#     )
#     print(len(events_paths))
#     # NOTE : around 3000~50000 events per frame, 37052336 in total 
#     # NOTE : 10000ごとに tracker 更新する？, 
#     # loading takes so much time
#     # event list の total events == 10000 くらい？を保持しておく　→ after processing, load next txt file
#     for file in events_paths:
#         data = np.loadtxt(file)
#         print(file)
#         data_list.append(data)
#     combined_data = np.concatenate(data_list, axis=0)
#     output_file = 'combined_data.txt'
#     np.savetxt(output_file, combined_data)
#     #print(combined_data)
#     #print(combined_data.shape)

def main():
    start_time = time.time()
    data = np.loadtxt('combined_data.txt')
    # print(data.shape)
    end_time = time.time()
    execution_time = end_time - start_time
    # NOTE : 150 seconds (room0) for loading all events
    # NOTE : loading a little at a time may be better?
    print("実行時間:", execution_time, "秒")


if __name__ == '__main__':
    main()