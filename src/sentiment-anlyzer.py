import pandas as pd
import openai
import time
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel
from typing import Literal

# Define structured output schemas
class EmotionScores(BaseModel):
    anger: float
    disgust: float  
    fear: float
    joy: float
    sadness: float
    surprise: float
    neutral: float

class EmotionResult(BaseModel):
    comment_index: int
    primary_emotion: Literal["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    confidence: float
    all_emotions: EmotionScores

class EmotionBatchResponse(BaseModel):
    results: List[EmotionResult]

class YouTubeEmotionAnalyzer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze_emotions_batch(self, comments: List[str], batch_size: int = 10) -> List[Dict]:
        """Analyze emotions for a batch of comments using GPT-5 with structured outputs"""
        results = []
        
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            
            # Create prompt for batch processing
            prompt = """Analyze the emotion of each Korean YouTube comment below. 
            
            Classify each comment into these emotions with confidence scores (0-1):
            - anger (분노): expressions of frustration, annoyance, or rage
            - disgust (혐오): expressions of revulsion, distaste, or aversion  
            - fear (두려움): expressions of anxiety, worry, or apprehension
            - joy (기쁨/즐거움): expressions of happiness, amusement, or pleasure (including ㅋㅋ, ㅎㅎ)
            - sadness (슬픔): expressions of sorrow, disappointment, or melancholy
            - surprise (놀람): expressions of astonishment, shock, or wonder
            - neutral (중립): balanced or emotionally neutral expressions
            
            For each comment, provide:
            1. The primary emotion (strongest emotion detected)
            2. Confidence score for the primary emotion (0-1)
            3. Scores for all emotions (must sum to approximately 1.0)
            
            Comments to analyze:"""
            
            for idx, comment in enumerate(batch):
                prompt += f"\n{idx}: {comment}"
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5",  # Updated to GPT-5
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert emotion analysis AI specializing in Korean text and social media language. Pay special attention to Korean internet slang, emoticons (ㅋㅋ, ㅎㅎ), and cultural context."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "emotion_analysis",
                            "strict": True,  # This is the structured output feature
                            "schema": EmotionBatchResponse.model_json_schema()
                        }
                    },
                    max_completion_tokens=3000,  # Updated parameter name for GPT-5
                    verbosity="medium",  # New GPT-5 parameter
                    reasoning_effort="medium"  # New GPT-5 parameter for better analysis
                )
                
                # Parse response using structured output
                response_text = response.choices[0].message.content.strip()
                
                try:
                    # GPT-5 with structured output should return properly formatted JSON
                    batch_data = json.loads(response_text)
                    batch_results = batch_data.get("results", [])
                    
                    # Adjust indices for global position and convert to dict format
                    for result in batch_results:
                        adjusted_result = {
                            'comment_index': i + result['comment_index'],
                            'primary_emotion': result['primary_emotion'],
                            'confidence': result['confidence'],
                            'all_emotions': result['all_emotions']
                        }
                        results.append(adjusted_result)
                    
                    print(f"✓ Successfully processed batch {i//batch_size + 1} with {len(batch_results)} results")
                    
                except json.JSONDecodeError as je:
                    print(f"JSON decode error for batch {i//batch_size + 1}: {je}")
                    # Fallback for malformed JSON
                    for j in range(len(batch)):
                        results.append({
                            'comment_index': i + j,
                            'primary_emotion': 'neutral',
                            'confidence': 0.5,
                            'all_emotions': {
                                'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0,
                                'sadness': 0, 'surprise': 0, 'neutral': 1.0
                            }
                        })
                
            except Exception as e:
                print(f"API error for batch {i//batch_size + 1}: {e}")
                # Add default results for failed batch
                for j in range(len(batch)):
                    results.append({
                        'comment_index': i + j,
                        'primary_emotion': 'neutral',
                        'confidence': 0.5,
                        'all_emotions': {
                            'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0,
                            'sadness': 0, 'surprise': 0, 'neutral': 1.0
                        }
                    })
            
            # Rate limiting - GPT-5 has higher rate limits but still good to be safe
            time.sleep(0.5)  # Reduced from 1 second
            print(f"Processed batch {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
        
        return results
    
    def analyze_emotions_single(self, comment: str) -> Dict:
        """Analyze single comment with GPT-5 thinking mode for complex cases"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Korean emotion analyst. Analyze this single comment deeply, considering cultural context, slang, and subtle emotional cues."
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze the emotion in this Korean comment: '{comment}'"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "single_emotion_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "primary_emotion": {
                                    "type": "string",
                                    "enum": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "all_emotions": {
                                    "type": "object",
                                    "properties": {
                                        "anger": {"type": "number", "minimum": 0, "maximum": 1},
                                        "disgust": {"type": "number", "minimum": 0, "maximum": 1},
                                        "fear": {"type": "number", "minimum": 0, "maximum": 1},
                                        "joy": {"type": "number", "minimum": 0, "maximum": 1},
                                        "sadness": {"type": "number", "minimum": 0, "maximum": 1},
                                        "surprise": {"type": "number", "minimum": 0, "maximum": 1},
                                        "neutral": {"type": "number", "minimum": 0, "maximum": 1}
                                    },
                                    "required": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
                                    "additionalProperties": False
                                },
                                "reasoning": {"type": "string"}
                            },
                            "required": ["primary_emotion", "confidence", "all_emotions", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.1,
                verbosity="high",  # More detailed analysis for single comments
                reasoning_effort="high"  # Use GPT-5's enhanced reasoning
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error analyzing single comment: {e}")
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {'neutral': 1.0, 'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0},
                'reasoning': 'Error occurred during analysis'
            }
    
    def analyze_csv(self, csv_file: str, output_file: str = None, use_detailed_analysis: bool = False) -> pd.DataFrame:
        """Analyze emotions for all comments in CSV file"""
        # Load data
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} comments")
        
        # Prepare comments for analysis
        comments = df['comment_text'].astype(str).tolist()
        
        # Choose analysis method
        if use_detailed_analysis and len(comments) <= 50:
            print("Using detailed single-comment analysis with GPT-5 thinking...")
            emotion_results = []
            for i, comment in enumerate(comments):
                result = self.analyze_emotions_single(comment)
                result['comment_index'] = i
                emotion_results.append(result)
                print(f"Processed comment {i+1}/{len(comments)}")
                time.sleep(0.3)
        else:
            print("Using batch analysis with GPT-5...")
            emotion_results = self.analyze_emotions_batch(comments)
        
        # Add results to dataframe with proper data types
        emotion_columns = ['primary_emotion', 'confidence', 'anger_score', 'disgust_score', 
                        'fear_score', 'joy_score', 'sadness_score', 'surprise_score', 'neutral_score']
        
        if use_detailed_analysis:
            emotion_columns.append('reasoning')
        
        # Initialize emotion columns with proper dtypes
        for col in emotion_columns:
            if col == 'primary_emotion' or col == 'reasoning':
                df[col] = pd.Series(dtype='string')
            else:
                df[col] = pd.Series(dtype='float64')  # Ensure numeric columns are float
        
        # Fill in results
        for result in emotion_results:
            idx = result['comment_index']
            if idx < len(df):
                df.loc[idx, 'primary_emotion'] = result['primary_emotion']
                # Explicitly convert to float
                df.loc[idx, 'confidence'] = float(result['confidence'])
                
                # Fill individual emotion scores as floats
                all_emotions = result.get('all_emotions', {})
                df.loc[idx, 'anger_score'] = float(all_emotions.get('anger', 0))
                df.loc[idx, 'disgust_score'] = float(all_emotions.get('disgust', 0))
                df.loc[idx, 'fear_score'] = float(all_emotions.get('fear', 0))
                df.loc[idx, 'joy_score'] = float(all_emotions.get('joy', 0))
                df.loc[idx, 'sadness_score'] = float(all_emotions.get('sadness', 0))
                df.loc[idx, 'surprise_score'] = float(all_emotions.get('surprise', 0))
                df.loc[idx, 'neutral_score'] = float(all_emotions.get('neutral', 0))
                
                if use_detailed_analysis:
                    df.loc[idx, 'reasoning'] = result.get('reasoning', '')
        
        # Verify data types before saving
        print("Data types after processing:")
        for col in emotion_columns:
            if col in df.columns:
                print(f"  {col}: {df[col].dtype}")
        
        # Save results
        if output_file is None:
            suffix = '_detailed_emotions' if use_detailed_analysis else '_with_emotions'
            output_file = csv_file.replace('.csv', f'{suffix}.csv')
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to: {output_file}")
        
        # Print summary
        self.print_emotion_summary(df)
        
        return df

        
    def print_emotion_summary(self, df: pd.DataFrame):
        """Print emotion analysis summary with GPT-5 enhanced insights"""
        print("\n=== GPT-5 Emotion Analysis Summary ===")
        emotion_counts = df['primary_emotion'].value_counts()
        print("Primary Emotions Distribution:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df)) * 100
            emoji_map = {
                'joy': '😊', 'anger': '😠', 'sadness': '😢', 
                'surprise': '😲', 'fear': '😨', 'disgust': '🤢', 'neutral': '😐'
            }
            emoji = emoji_map.get(emotion, '🤔')
            print(f"  {emoji} {emotion}: {count} ({percentage:.1f}%)")
        
        # Convert confidence column to numeric and handle NaN values
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        
        # Filter out rows where confidence is NaN
        valid_confidence_df = df.dropna(subset=['confidence'])
        
        if len(valid_confidence_df) > 0:
            print(f"\nAverage Confidence: {valid_confidence_df['confidence'].mean():.2f}")
            print(f"High Confidence Predictions (>0.8): {len(valid_confidence_df[valid_confidence_df['confidence'] > 0.8])}/{len(valid_confidence_df)} ({len(valid_confidence_df[valid_confidence_df['confidence'] > 0.8])/len(valid_confidence_df)*100:.1f}%)")
            
            # Most emotional comments - using valid confidence data only
            print(f"\nMost Confident Predictions:")
            if len(valid_confidence_df) >= 3:
                top_confident = valid_confidence_df.nlargest(3, 'confidence')[['comment_text', 'primary_emotion', 'confidence']]
            else:
                top_confident = valid_confidence_df.nlargest(len(valid_confidence_df), 'confidence')[['comment_text', 'primary_emotion', 'confidence']]
                
            for _, row in top_confident.iterrows():
                emoji = {'joy': '😊', 'anger': '😠', 'sadness': '😢', 'surprise': '😲', 'fear': '😨', 'disgust': '🤢', 'neutral': '😐'}.get(row['primary_emotion'], '🤔')
                print(f"  {emoji} [{row['primary_emotion']}] {row['confidence']:.3f}: {row['comment_text'][:80]}...")
        else:
            print("\nNo valid confidence scores found.")
        
        # Emotion intensity analysis - also convert emotion scores to numeric
        print(f"\nEmotion Intensity Analysis:")
        for emotion in ['joy', 'anger', 'sadness']:
            score_col = f'{emotion}_score'
            if score_col in df.columns:
                # Convert to numeric
                df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
                valid_scores = df[score_col].dropna()
                if len(valid_scores) > 0:
                    avg_score = valid_scores.mean()
                    max_score = valid_scores.max()
                    print(f"  {emotion}: avg={avg_score:.3f}, max={max_score:.3f}")


# Enhanced usage with GPT-5 features
def main():
    # Set your OpenAI API key
    API_KEY = "your-openai-api-key"
    
    analyzer = YouTubeEmotionAnalyzer()
    
    # Choose analysis type
    print("Choose analysis type:")
    print("1. Fast batch analysis (recommended for >50 comments)")
    print("2. Detailed analysis with reasoning (recommended for ≤50 comments)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    use_detailed = choice == "2"
    
    # Analyze the CSV file
    result_df = analyzer.analyze_csv(
        'outputs/comments/sample.csv', 
        'sample_with_gpt5_emotions.csv',
        use_detailed_analysis=use_detailed
    )
    
    return result_df

if __name__ == "__main__":
    main()
