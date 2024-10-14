import os

def save_folder_names_to_txt(directory_path, output_txt_file):
    # 해당 경로의 폴더 목록을 가져옴
    folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    
    # 파일에 저장
    with open(output_txt_file, 'w') as f:
        for folder in folder_names:
            f.write(f"{folder}\n")
    
    print(f"폴더 목록이 {output_txt_file}에 저장되었습니다.")

# 경로와 출력 파일 지정
directory_path = '/root/dataset/WOD/validation/sequences'  # 폴더들이 있는 경로
output_txt_file = 'val_wod.txt'  # 출력될 텍스트 파일 이름

# 함수 실행
save_folder_names_to_txt(directory_path, output_txt_file)