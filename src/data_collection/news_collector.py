"""
News Data Collector Module
Collects news articles from multiple free sources using web scraping
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time
import re
from urllib.parse import quote

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Collects news articles from multiple free sources.
    
    Sources:
    - Yahoo Finance (web scraping)
    - Google News RSS (RSS feeds)
    - Finviz (web scraping)
    
    All sources are 100% free with no API keys required.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize the news collector.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'TSLA', 'NVDA'])
        """
        self.symbols = symbols
        self.news_data = []
        
        # Headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Initialized NewsCollector for {len(symbols)} stocks: {', '.join(symbols)}")
    
    def collect_yahoo_finance_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Scrape news articles from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of dictionaries containing news data
        """
        articles = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            logger.info(f"Fetching Yahoo Finance news for {symbol}...")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles (Yahoo Finance structure may change)
            news_items = soup.find_all('h3', class_=re.compile('Mb'))
            
            for item in news_items[:max_articles]:
                try:
                    # Extract title
                    title_elem = item.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    
                    # Make link absolute if relative
                    if link.startswith('/'):
                        link = f"https://finance.yahoo.com{link}"
                    
                    # Try to find publication date
                    date_elem = item.find_next('div', class_=re.compile('C'))
                    pub_date = datetime.now()  # Default to now if not found
                    
                    if date_elem:
                        date_text = date_elem.get_text(strip=True)
                        # Parse relative dates like "2 hours ago", "1 day ago"
                        pub_date = self._parse_relative_date(date_text)
                    
                    article = {
                        'symbol': symbol,
                        'title': title,
                        'url': link,
                        'published_date': pub_date,
                        'source': 'Yahoo Finance',
                        'text': title  # Use title as text for sentiment analysis
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.debug(f"Error parsing article: {str(e)}")
                    continue
            
            logger.info(f"✅ Collected {len(articles)} articles from Yahoo Finance for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
        
        return articles
    
    def collect_google_news_rss(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Collect news from Google News RSS feeds.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of dictionaries containing news data
        """
        articles = []
        
        try:
            # Create search query
            query = f"{symbol} stock"
            encoded_query = quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            logger.info(f"Fetching Google News RSS for {symbol}...")
            
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Extract article data
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Extract summary/description
                    summary = entry.get('summary', title)
                    # Remove HTML tags from summary
                    summary = BeautifulSoup(summary, 'html.parser').get_text(strip=True)
                    
                    article = {
                        'symbol': symbol,
                        'title': title,
                        'url': link,
                        'published_date': pub_date,
                        'source': 'Google News',
                        'text': f"{title}. {summary}"
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.debug(f"Error parsing RSS entry: {str(e)}")
                    continue
            
            logger.info(f"✅ Collected {len(articles)} articles from Google News for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching Google News for {symbol}: {str(e)}")
        
        return articles
    
    def collect_finviz_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """
        Scrape news from Finviz.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of dictionaries containing news data
        """
        articles = []
        
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            logger.info(f"Fetching Finviz news for {symbol}...")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news table
            news_table = soup.find('table', class_='fullview-news-outer')
            
            if news_table:
                rows = news_table.find_all('tr')
                
                for row in rows[:max_articles]:
                    try:
                        # Extract date/time
                        date_cell = row.find('td', align='right')
                        title_cell = row.find('a', class_='tab-link-news')
                        
                        if not title_cell:
                            continue
                        
                        title = title_cell.get_text(strip=True)
                        link = title_cell.get('href', '')
                        
                        # Parse date
                        pub_date = datetime.now()
                        if date_cell:
                            date_text = date_cell.get_text(strip=True)
                            pub_date = self._parse_finviz_date(date_text)
                        
                        article = {
                            'symbol': symbol,
                            'title': title,
                            'url': link,
                            'published_date': pub_date,
                            'source': 'Finviz',
                            'text': title
                        }
                        
                        articles.append(article)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing Finviz article: {str(e)}")
                        continue
            
            logger.info(f"✅ Collected {len(articles)} articles from Finviz for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching Finviz news for {symbol}: {str(e)}")
        
        return articles
    
    def collect_all_news(self, max_articles_per_source: int = 20) -> pd.DataFrame:
        """
        Collect news from all sources for all symbols.
        
        Args:
            max_articles_per_source: Maximum articles per source per symbol
            
        Returns:
            DataFrame with all collected news
        """
        logger.info(f"Starting news collection for {len(self.symbols)} stocks...")
        
        all_articles = []
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting news for {symbol}")
            logger.info(f"{'='*60}")
            
            # Collect from Yahoo Finance
            yahoo_articles = self.collect_yahoo_finance_news(symbol, max_articles_per_source)
            all_articles.extend(yahoo_articles)
            time.sleep(1)  # Be respectful with requests
            
            # Collect from Google News
            google_articles = self.collect_google_news_rss(symbol, max_articles_per_source)
            all_articles.extend(google_articles)
            time.sleep(1)
            
            # Collect from Finviz
            finviz_articles = self.collect_finviz_news(symbol, max_articles_per_source)
            all_articles.extend(finviz_articles)
            time.sleep(1)
            
            logger.info(f"Total articles for {symbol}: {len(yahoo_articles) + len(google_articles) + len(finviz_articles)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Remove duplicates based on title
            df = df.drop_duplicates(subset=['title'], keep='first')
            
            # Sort by date
            df = df.sort_values('published_date', ascending=False)
            
            # Reset index
            df = df.reset_index(drop=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ News collection complete!")
        logger.info(f"Total unique articles: {len(df)}")
        logger.info(f"{'='*60}\n")
        
        self.news_data = df
        return df
    
    def _parse_relative_date(self, date_text: str) -> datetime:
        """
        Parse relative date strings like '2 hours ago', '1 day ago'.
        
        Args:
            date_text: Relative date string
            
        Returns:
            datetime object
        """
        now = datetime.now()
        date_text = date_text.lower()
        
        try:
            if 'hour' in date_text or 'hr' in date_text:
                hours = int(re.search(r'\d+', date_text).group())
                return now - timedelta(hours=hours)
            elif 'minute' in date_text or 'min' in date_text:
                minutes = int(re.search(r'\d+', date_text).group())
                return now - timedelta(minutes=minutes)
            elif 'day' in date_text:
                days = int(re.search(r'\d+', date_text).group())
                return now - timedelta(days=days)
            elif 'week' in date_text:
                weeks = int(re.search(r'\d+', date_text).group())
                return now - timedelta(weeks=weeks)
            elif 'month' in date_text:
                months = int(re.search(r'\d+', date_text).group())
                return now - timedelta(days=months*30)
        except:
            pass
        
        return now
    
    def _parse_finviz_date(self, date_text: str) -> datetime:
        """
        Parse Finviz date format.
        
        Args:
            date_text: Date string from Finviz
            
        Returns:
            datetime object
        """
        try:
            # Finviz uses formats like "Dec-03-24 09:30AM" or "Today 09:30AM"
            if 'today' in date_text.lower():
                time_part = date_text.split()[-1]
                # Parse time
                return datetime.now().replace(
                    hour=int(time_part.split(':')[0]),
                    minute=int(time_part.split(':')[1][:2])
                )
            else:
                # Try to parse full date
                return datetime.strptime(date_text, "%b-%d-%y %I:%M%p")
        except:
            return datetime.now()
    
    def save_news(self, output_dir: Path):
        """
        Save collected news to CSV.
        
        Args:
            output_dir: Directory to save news data
        """
        if self.news_data is None or len(self.news_data) == 0:
            logger.warning("No news data to save")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all news
        filepath = output_dir / "all_news.csv"
        self.news_data.to_csv(filepath, index=False)
        logger.info(f"💾 Saved all news to {filepath}")
        
        # Save per symbol
        for symbol in self.symbols:
            symbol_news = self.news_data[self.news_data['symbol'] == symbol]
            if len(symbol_news) > 0:
                symbol_filepath = output_dir / f"{symbol}_news.csv"
                symbol_news.to_csv(symbol_filepath, index=False)
                logger.info(f"💾 Saved {symbol} news to {symbol_filepath}")
    
    def get_news_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for collected news.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.news_data is None or len(self.news_data) == 0:
            return pd.DataFrame()
        
        summaries = []
        
        for symbol in self.symbols:
            symbol_news = self.news_data[self.news_data['symbol'] == symbol]
            
            summary = {
                'Symbol': symbol,
                'Total_Articles': len(symbol_news),
                'Yahoo_Finance': len(symbol_news[symbol_news['source'] == 'Yahoo Finance']),
                'Google_News': len(symbol_news[symbol_news['source'] == 'Google News']),
                'Finviz': len(symbol_news[symbol_news['source'] == 'Finviz']),
                'Date_Range': f"{symbol_news['published_date'].min()} to {symbol_news['published_date'].max()}"
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def main():
    """
    Main function to demonstrate usage.
    """
    from src.utils.config import STOCK_SYMBOLS, RAW_DATA_DIR
    
    # Initialize collector
    collector = NewsCollector(symbols=STOCK_SYMBOLS)
    
    # Collect news
    news_df = collector.collect_all_news(max_articles_per_source=15)
    
    # Print summary
    print("\n" + "="*80)
    print("NEWS COLLECTION SUMMARY")
    print("="*80)
    summary = collector.get_news_summary()
    print(summary.to_string(index=False))
    print("="*80 + "\n")
    
    # Save news
    collector.save_news(RAW_DATA_DIR)
    
    print(f"\n✅ News collection complete!")
    print(f"📁 Data saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
