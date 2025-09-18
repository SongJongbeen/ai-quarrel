import csv
from crawler import ytCrawler
from wclouder import ytWclouder
from emotionAnalyzer import ytEmotionAnalyzer

# with open('data/videos.csv', 'r', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         idx = row[0]
#         url = row[4]

#         crawler = ytCrawler()
#         output_file = f"outputs/youtube_comments_{idx}.csv"

#         if not output_file:
#             output_file = None

#         crawler.crawl_youtube_comments(url, output_file)

# for idx in range(1, 13):
#     generator = ytWclouder(f"outputs/youtube_comments_{idx}.csv")
#     if generator.load_data():
#         generator.process_comments()
#         generator.print_statistics()

#         wordcloud = generator.create_wordcloud(
#             width=1200, 
#             height=600, 
#             max_words=150,
#             background_color='white',
#             colormap='plasma'  # Try 'viridis', 'plasma', 'inferno', 'magma'
#         )

#         generator.display_wordcloud(
#             wordcloud,
#             title=f"YouTube Comments Word Cloud {idx}",
#             save_path=f"outputs/youtube_wordcloud_{idx}.png"
#         )

#         generator.save_statistics(f"outputs/youtube_statistics_{idx}.txt")

for idx in range(1, 13):
    if idx != 12:
        continue

    analyzer = ytEmotionAnalyzer()

    choice = "1"
    use_detailed = choice == "2"

    result_df = analyzer.analyze_csv(
        f"outputs/comments/youtube_comments_{idx}.csv",
        f"outputs/emotions/youtube_emotion_{idx}.csv",
        use_detailed_analysis=use_detailed
    )
