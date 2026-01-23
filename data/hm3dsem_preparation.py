import os
import shutil
from glob import glob

def build_hovsg_hm3d_structure(src, dst):
    """
    Habitat Matterport 3D Semantics Dataset(HM3D) preparation into HOV-SG format.

    origin: path to minival or val containing:
    - val/
        hm3d-val-glb-v0.2
        hm3d-val-habitat-v0.2
        hm3d-val-semantic-annots-v0.2
        hm3d-val-semantic-configs-v0.2

    HOV-SG format:
    - hm3d_annotated_basis.scene_dataset_config.json
        - val/
            scene_id/
                scene_id.glb
                scene_id.basis.glb
                scene_id.basis.navmesh
                scene_id.semantic.glb
                scene_id.semantic.txt
            scene_id/
                ...
    """

    glb_dir      = glob(os.path.join(src, "*glb-v0.2"))[0]
    habitat_dir  = glob(os.path.join(src, "*habitat-v0.2"))[0]
    annot_dir    = glob(os.path.join(src, "*semantic-annots-v0.2"))[0]
    configs_dir  = glob(os.path.join(src, "*semantic-configs-v0.2"))[0]

    print("ROOT", src)
    print("GLB:", glb_dir)
    print("HABITAT:", habitat_dir)
    print("ANNOTS:", annot_dir)
    print("CONFIGS:", configs_dir)

    # HOV-SG format directory
    os.makedirs(dst, exist_ok=True)
    val_root = os.path.join(dst, "val")
    os.makedirs(val_root, exist_ok=True)

    # Copy annotation config json
    for json_file in glob(os.path.join(configs_dir, "*.json")):
        shutil.copy(json_file, os.path.join(dst, os.path.basename(json_file)))

    # Find all available scene IDs
    scene_ids = sorted(os.listdir(glb_dir))
    annotated_scene_ids = set(os.listdir(annot_dir))

    for scene_id in scene_ids:
        
        # 숨김 파일 무시
        if scene_id.startswith("."):
            continue

        scene_folder_glb      = os.path.join(glb_dir, scene_id)
        scene_folder_habitat  = os.path.join(habitat_dir, scene_id)
        scene_folder_annnnots = os.path.join(annot_dir, scene_id)

        # 디렉터리가 아닌 경우 무시
        if not (os.path.isdir(scene_folder_glb)
                and os.path.isdir(scene_folder_habitat)
                and os.path.isdir(scene_folder_annnnots)):
            continue
        
        # semantic-annots에 없는 scene은 생략
        if scene_id not in annotated_scene_ids:
            continue
        
        scene_folder_glb = os.path.join(glb_dir, scene_id)
        scene_folder_habitat = os.path.join(habitat_dir, scene_id)
        scene_folder_annnnots = os.path.join(annot_dir, scene_id)

        print(f"Processing scene {scene_id}")

        out_scene_dir = os.path.join(val_root, scene_id)
        os.makedirs(out_scene_dir, exist_ok=True)

        # Scene Code 추출
        scene_code = scene_id.split("-")[-1]
        print(f"  Scene code: {scene_code}")
        
        # GLB files
        glb = os.path.join(scene_folder_glb, f"{scene_code}.glb")
        shutil.copy(glb, os.path.join(out_scene_dir, f"{scene_code}.glb"))

        # BASIS + NAVMESH
        basis_glb = os.path.join(scene_folder_habitat, f"{scene_code}.basis.glb")
        navmesh   = os.path.join(scene_folder_habitat, f"{scene_code}.basis.navmesh")
        shutil.copy(basis_glb, os.path.join(out_scene_dir, f"{scene_code}.basis.glb"))
        shutil.copy(navmesh, os.path.join(out_scene_dir, f"{scene_code}.basis.navmesh"))
    
        # semantic glb + txt
        sem_glb = os.path.join(scene_folder_annnnots, f"{scene_code}.semantic.glb")
        sem_txt = os.path.join(scene_folder_annnnots, f"{scene_code}.semantic.txt")
        shutil.copy(sem_glb, os.path.join(out_scene_dir, f"{scene_code}.semantic.glb"))
        shutil.copy(sem_txt, os.path.join(out_scene_dir, f"{scene_code}.semantic.txt"))

if __name__ == "__main__":
    # hm3d dataset path
    src_root = "./hm3dsem-origin/val"
    dst_root = "./hm3dsem/val"

    print(f"Source root : {src_root}")
    print(f"Target root : {dst_root}")
    build_hovsg_hm3d_structure(src_root, dst_root)