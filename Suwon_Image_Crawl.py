from icrawler.builtin import GoogleImageCrawler

storage = {'root_dir': 'suwon_budget_images'}

crawler = GoogleImageCrawler(storage=storage)

crawler.crawl(
    keyword='수원시 주민참여예산 사진', 
    max_num=500,              
    min_size=(200,200),       
    file_idx_offset=0   
)
