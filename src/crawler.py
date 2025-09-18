import csv
import pandas as pd
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import re
from urllib.parse import urlparse, parse_qs

class ytCrawler:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url):
        """Extract the video ID from a YouTube URL"""
        parsed_url = urlparse(url)

        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            elif parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]

        return None

    def get_video_comments(self, video_id):
        """Fetch all comments and replies from a YouTube video"""
        comments_data = []

        try:
            # Get comment threads
            request = self.youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                order='relevance'
            )
            response = request.execute()

            while response:
                for item in response['items']:
                    # Top-level comment
                    top_comment = item['snippet']['topLevelComment']['snippet']

                    comment_data = {
                        'comment_id': item['snippet']['topLevelComment']['id'],
                        'video_id': video_id,
                        'comment_type': 'main_comment',
                        'parent_id': '',
                        'author_name': top_comment['authorDisplayName'],
                        'author_channel_url': top_comment.get('authorChannelUrl', ''),
                        'comment_text': top_comment['textDisplay'],
                        'like_count': top_comment['likeCount'],
                        'published_at': top_comment['publishedAt'],
                        'updated_at': top_comment.get('updatedAt', ''),
                        'reply_count': item['snippet']['totalReplyCount']
                    }

                    comments_data.append(comment_data)

                    # Check for replies
                    if 'replies' in item:
                        for reply in item['replies']['comments']:
                            reply_snippet = reply['snippet']

                            reply_data = {
                                'comment_id': reply['id'],
                                'video_id': video_id,
                                'comment_type': 'reply',
                                'parent_id': item['snippet']['topLevelComment']['id'],
                                'author_name': reply_snippet['authorDisplayName'],
                                'author_channel_url': reply_snippet.get('authorChannelUrl', ''),
                                'comment_text': reply_snippet['textDisplay'],
                                'like_count': reply_snippet['likeCount'],
                                'published_at': reply_snippet['publishedAt'],
                                'updated_at': reply_snippet.get('updatedAt', ''),
                                'reply_count': 0
                            }

                            comments_data.append(reply_data)

                # Check for next page
                if 'nextPageToken' in response:
                    request = self.youtube.commentThreads().list(
                        part='snippet,replies',
                        videoId=video_id,
                        maxResults=100,
                        pageToken=response['nextPageToken'],
                        order='relevance'
                    )
                    response = request.execute()
                else:
                    break

        except Exception as e:
            print(f"Error fetching comments: {str(e)}")

        return comments_data

    def save_to_csv(self, comments_data, filename):
        """Save comments data to CSV file"""
        if not comments_data:
            print("No comments data to save.")
            return

        df = pd.DataFrame(comments_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Comments saved to {filename}")
        print(f"Total comments and replies: {len(comments_data)}")

        # Print summary
        main_comments = len(df[df['comment_type'] == 'main_comment'])
        replies = len(df[df['comment_type'] == 'reply'])
        print(f"Main comments: {main_comments}")
        print(f"Replies: {replies}")

    def crawl_youtube_comments(self, url, output_filename=None):
        """Main method to crawl comments from YouTube URL"""
        # Extract video ID
        video_id = self.extract_video_id(url)
        if not video_id:
            print("Invalid YouTube URL")
            return

        print(f"Extracting comments for video ID: {video_id}")

        # Get comments
        comments_data = self.get_video_comments(video_id)

        # Generate filename if not provided
        if not output_filename:
            output_filename = f"youtube_comments_{video_id}.csv"

        # Save to CSV
        self.save_to_csv(comments_data, output_filename)

# Usage example
def main():
    # Initialize crawler
    crawler = ytCrawler()

    # Get YouTube URL from user
    # youtube_url = input("Enter YouTube video URL: ")
    youtube_url = "https://www.youtube.com/watch?v=Zqzb7kE4dLI"

    # Optional: specify output filename
    # output_file = input("Enter output filename (press Enter for auto-generated): ").strip()
    output_file = "outputs/youtube_comments_Zqzb7kE4dLI.csv"

    if not output_file:
        output_file = None
        
    # Crawl comments
    crawler.crawl_youtube_comments(youtube_url, output_file)

if __name__ == "__main__":
    main()
