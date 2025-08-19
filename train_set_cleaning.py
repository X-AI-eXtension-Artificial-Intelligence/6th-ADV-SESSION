import os
import shutil

base_dir = r"C:\상품 이미지\Training"

def step1_keep_only_s1_and_meta():
    """상품 폴더 내에서 s_1 또는 s_1_meta로 끝나지 않는 파일 삭제"""
    delete_count = 0
    for category_folder in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_folder)
        if not os.path.isdir(category_path):
            continue

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if not os.path.isdir(product_path):
                continue

            for file in os.listdir(product_path):
                file_path = os.path.join(product_path, file)
                if not os.path.isfile(file_path):
                    continue

                filename_no_ext = os.path.splitext(file)[0]
                if not (filename_no_ext.endswith("s_1") or filename_no_ext.endswith("s_1_meta")):
                    os.remove(file_path)
                    delete_count += 1
    print(f"[1단계] {delete_count}개 파일 삭제 완료")


def step2_remove_empty_folders():
    """비어있는 상품 폴더 삭제"""
    empty_folder_count = 0
    for category_folder in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_folder)
        if not os.path.isdir(category_path):
            continue

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if not os.path.isdir(product_path):
                continue

            if not os.listdir(product_path):
                os.rmdir(product_path)
                empty_folder_count += 1
    print(f"[2단계] {empty_folder_count}개 빈 폴더 삭제 완료")


def step3_angle_priority():
    """각도 우선순위 선택 (30 → 00 → 60), 나머지 파일 삭제"""
    total_keep, total_delete = 0, 0
    for category_folder in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_folder)
        if not os.path.isdir(category_path):
            continue

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if not os.path.isdir(product_path):
                continue

            files = os.listdir(product_path)
            filenames_no_ext = [os.path.splitext(f)[0] for f in files]

            if any("30_s_1" in fn for fn in filenames_no_ext):
                chosen_prefix = "30_s_1"
            elif any("00_s_1" in fn for fn in filenames_no_ext):
                chosen_prefix = "00_s_1"
            elif any("60_s_1" in fn for fn in filenames_no_ext):
                chosen_prefix = "60_s_1"
            else:
                chosen_prefix = None

            if chosen_prefix:
                for file in files:
                    file_no_ext, ext = os.path.splitext(file)
                    if chosen_prefix in file_no_ext:
                        total_keep += 1
                    else:
                        os.remove(os.path.join(product_path, file))
                        total_delete += 1
            else:
                for file in files:
                    os.remove(os.path.join(product_path, file))
                    total_delete += 1

    print(f"[3단계] 유지 파일 {total_keep}개, 삭제 파일 {total_delete}개")


def step4_keep_only_meta_in_label():
    """[라벨] 폴더 안에서는 meta만 유지"""
    total_keep, total_delete = 0, 0
    for category_folder in os.listdir(base_dir):
        if not category_folder.startswith("[라벨]"):
            continue
        category_path = os.path.join(base_dir, category_folder)

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if not os.path.isdir(product_path):
                continue

            for file in os.listdir(product_path):
                file_no_ext, ext = os.path.splitext(file)
                if file_no_ext.endswith("meta"):
                    total_keep += 1
                else:
                    os.remove(os.path.join(product_path, file))
                    total_delete += 1
    print(f"[4단계] [라벨] 유지 {total_keep}개, 삭제 {total_delete}개")


def step5_remove_unmatched_source():
    """[원천] 폴더 중 [라벨]에 없는 상품 폴더 삭제"""
    label_products = set()
    for category_folder in os.listdir(base_dir):
        if not category_folder.startswith("[라벨]"):
            continue
        category_path = os.path.join(base_dir, category_folder)

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if os.path.isdir(product_path):
                label_products.add(product_folder)

    delete_candidates = []
    for category_folder in os.listdir(base_dir):
        if not category_folder.startswith("[원천]"):
            continue
        category_path = os.path.join(base_dir, category_folder)

        for product_folder in os.listdir(category_path):
            product_path = os.path.join(category_path, product_folder)
            if os.path.isdir(product_path) and product_folder not in label_products:
                shutil.rmtree(product_path)  # 폴더 전체 삭제
                delete_candidates.append(product_path)

    print(f"[5단계] [원천] 불일치 상품 폴더 {len(delete_candidates)}개 삭제")


if __name__ == "__main__":
    print("=== Preprocessing 시작 ===")
    step1_keep_only_s1_and_meta()
    step2_remove_empty_folders()
    step3_angle_priority()
    step4_keep_only_meta_in_label()
    step5_remove_unmatched_source()
    print("=== Preprocessing 완료 ===")
