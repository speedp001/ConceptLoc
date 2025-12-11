import os
import hydra
from omegaconf import DictConfig
from hovsg.graph.graph import Graph

# pylint: disable=all

# ./config/create_graph.yaml 파일에서 설정 불러오기
'''
Hydra가 실제로 하는 일
def _wrapped_main():

    cfg_dict = yaml_load("../config/create_graph.yaml")
    params = DictConfig(cfg_dict)
    
    return main(params)
'''
@hydra.main(version_base=None, config_path="./config", config_name="create_graph")
def main(params: DictConfig):
    # create logging directory
    # 저장 디렉토리 설정
    save_dir = os.path.join(params.main.save_path, params.main.scene_id)
    params.main.save_path = save_dir
    params.main.dataset_path = os.path.join(params.main.dataset_path, params.main.split, params.main.scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # create graph
    hovsg = Graph(params)
    hovsg.create_feature_map() # create feature map

    # save full point cloud, features, and masked point clouds (pcd for all objects)
    hovsg.save_masked_pcds(path=save_dir, state="both")
    hovsg.save_full_pcd(path=save_dir)
    hovsg.save_full_pcd_feats(path=save_dir)
    
    # for debugging: load preconstructed map as follows
    # hovsg.load_full_pcd(path=save_dir)
    # hovsg.load_full_pcd_feats(path=save_dir)
    # hovsg.load_masked_pcds(path=save_dir)

    # create graph, only if dataset is hm3dsem
    print(params.main.dataset)

    hovsg.build_graph(save_path=save_dir)

if __name__ == "__main__":
    main()