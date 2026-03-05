THRESHOLD = 30
MIN_AREA = 100


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    # TODO: 이미지 로드 → 엣지 검출 → 이진화 → Connected Component → 최소 면적 필터
