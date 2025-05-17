from icrawler.builtin import GoogleImageCrawler

# 1) 저장 폴더 지정
storage = {'root_dir': 'suwon_budget_images'}

# 2) 크롤러 인스턴스 생성
crawler = GoogleImageCrawler(storage=storage)

# 3) 크롤링 실행
crawler.crawl(
    keyword='수원시 주민참여예산 사진',  # 검색 키워드
    max_num=500,                      # 최대 이미지 개수 (원하는 만큼 조정)
    min_size=(200,200),               # 최소 이미지 해상도 필터 (옵션)
    file_idx_offset=0                 # 파일명 번호 시작값
)
