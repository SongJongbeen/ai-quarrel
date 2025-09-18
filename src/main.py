import csv
from crawler import ytCrawler

with open('data/videos.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        idx = row[0]
        url = row[4]
        
        crawler = ytCrawler()
        output_file = f"outputs/youtube_comments_{idx}.csv"

        if not output_file:
            output_file = None

        crawler.crawl_youtube_comments(url, output_file)
