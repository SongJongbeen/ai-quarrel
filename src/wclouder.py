import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from collections import Counter
import matplotlib.font_manager as fm

class ytWclouder:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.processed_text = ""

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def clean_text(self, text):
        if pd.isna(text):
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove @ mentions
        text = re.sub(r'@[A-Za-z0-9_-]+', '', text)
        
        # Remove timestamps (like 2:33)
        text = re.sub(r'\d+:\d+', '', text)
        
        # Remove excessive ㅋ, ㅎ laughing patterns (but keep some)
        text = re.sub(r'ㅋ{3,}', 'ㅋㅋ', text)
        text = re.sub(r'ㅎ{3,}', 'ㅎㅎ', text)
        
        # Remove special characters but keep Korean, numbers, and basic punctuation
        text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_comments(self):
        """Process all comments and replies to create combined text"""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return

        # Combine all comment texts
        all_comments = []

        for _, row in self.df.iterrows():
            cleaned_comment = self.clean_text(row['comment_text'])
            if cleaned_comment:
                # Weight comments by like count (minimum weight of 1)
                weight = max(1, int(row['like_count']) // 10 + 1)
                all_comments.extend([cleaned_comment] * weight)

        self.processed_text = ' '.join(all_comments)
        print(f"Processed {len(all_comments)} weighted comments")

    def get_korean_font_path(self):
        """Find Korean font path for proper display"""
        
        # First, try to find font files by path
        possible_font_paths = [
            # Windows paths
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/malgunbd.ttf', 
            'C:/Windows/Fonts/gulim.ttf',
            'C:/Windows/Fonts/batang.ttf',
            # Mac paths
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/NanumGothic.ttf',
            '/Library/Fonts/NanumBarunGothic.ttf',
            # Linux paths
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        ]
        
        # Check if any of these font files exist
        for font_path in possible_font_paths:
            if os.path.exists(font_path):
                print(f"Found font at: {font_path}")
                return font_path

        # If no direct paths work, try to find via matplotlib font manager
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        
        # Look for fonts that support Korean
        korean_font_keywords = ['Noto', 'Nanum', 'Malgun', 'Gothic', 'CJK']
        
        for font_path in font_list:
            font_name = Path(font_path).stem.lower()
            if any(keyword.lower() in font_name for keyword in korean_font_keywords):
                print(f"Found Korean-compatible font: {font_path}")
                return font_path
        
        # Last resort - return None to use default font
        print("No Korean font found, using default font")
        return None


    def create_wordcloud(self, width=1200, height=600, max_words=200, 
                        background_color='white', colormap='viridis'):
        """Create word cloud from processed comments"""
        if not self.processed_text:
            print("No processed text available. Please run process_comments() first.")
            return None

        # Get Korean font
        font_path = self.get_korean_font_path()
        
        # Configure WordCloud
        wordcloud_config = {
            'width': width,
            'height': height,
            'max_words': max_words,
            'background_color': background_color,
            'colormap': colormap,
            'relative_scaling': 0.5,
            'min_font_size': 10,
            'max_font_size': 100,
            'random_state': 42
        }
        
        if font_path:
            wordcloud_config['font_path'] = font_path

        print(f"wordcloud_config: {wordcloud_config}")
        print(f"wordcloud_config['font_path']: {wordcloud_config['font_path']}")
        print(f"processed_text: {self.processed_text}")

        # Create WordCloud
        try:
            wordcloud = WordCloud(**wordcloud_config).generate(self.processed_text)
            return wordcloud
        except Exception as e:
            print(f"Error creating word cloud: {e}")
            return None

    def display_wordcloud(self, wordcloud, title="YouTube Comments Word Cloud", 
                         figsize=(15, 8), save_path=None):
        """Display and optionally save the word cloud"""
        if wordcloud is None:
            print("No word cloud to display.")
            return

        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Word cloud saved to: {save_path}")
        
        plt.show()

    def get_word_frequency(self, top_n=20):
        """Get most frequent words"""
        if not self.processed_text:
            return {}
        
        words = self.processed_text.split()

        # TODO: Set rules for filtering words
        # Filter out single characters and common words
        filtered_words = [word for word in words if len(word) > 1 and word not in ['이거', '그거', '저거', '근데', '그냥']]
        
        word_freq = Counter(filtered_words)
        return dict(word_freq.most_common(top_n))
    
    def print_statistics(self):
        """Print statistics about the comments"""
        if self.df is None:
            return
        
        main_comments = len(self.df[self.df['comment_type'] == 'main_comment'])
        replies = len(self.df[self.df['comment_type'] == 'reply'])
        total_likes = self.df['like_count'].sum()
        
        print("\n=== Comment Statistics ===")
        print(f"Main comments: {main_comments}")
        print(f"Replies: {replies}")
        print(f"Total comments: {len(self.df)}")
        print(f"Total likes: {total_likes}")
        print(f"Average likes per comment: {total_likes/len(self.df):.1f}")

        # Most liked comment
        most_liked = self.df.loc[self.df['like_count'].idxmax()]
        print(f"\nMost liked comment ({most_liked['like_count']} likes):")
        print(f"'{self.clean_text(most_liked['comment_text'])}'")

        return f"Main comments: {main_comments}\nReplies: {replies}\nTotal comments: {len(self.df)}\nTotal likes: {total_likes}\nAverage likes per comment: {total_likes/len(self.df):.1f}\nMost liked comment: {self.clean_text(most_liked['comment_text'])}"

    def save_statistics(self, save_path):
        """Save the statistics to a file"""
        with open(save_path, 'w', encoding='utf-8') as file:
            contents = self.print_statistics() + "\n===Top 15 Words===\n"
            word_freq = self.get_word_frequency(15)
            for word, count in word_freq.items():
                contents += f"{word}: {count}\n"
            file.write(contents)

        print(f"Statistics saved to: {save_path}")


def main():
    # Initialize the word cloud generator
    generator = ytWclouder('outputs/sample.csv')
    
    # Load and process data
    if generator.load_data():
        generator.process_comments()
        
        # Print statistics
        generator.print_statistics()
        
        # Create word cloud
        wordcloud = generator.create_wordcloud(
            width=1200, 
            height=600, 
            max_words=150,
            background_color='white',
            colormap='plasma'  # Try 'viridis', 'plasma', 'inferno', 'magma'
        )
        
        # Display word cloud
        generator.display_wordcloud(
            wordcloud, 
            title="YouTube Comments Word Cloud",
            save_path="youtube_wordcloud.png"
        )

        # Save word cloud
        generator.save_statistics("youtube_statistics.txt")

if __name__ == "__main__":
    main()
