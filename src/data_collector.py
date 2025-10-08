"""
Data collection module for Japanese sport sponsor logos
"""
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import numpy as np
from PIL import Image
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JapaneseSportLogoCollector:
    """
    Collects Japanese sport sponsor logos from various sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.setup_selenium()
        
    def setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content scraping"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def collect_from_websites(self) -> List[Dict]:
        """
        Collect logos from Japanese sport websites
        """
        websites = [
            "https://npb.jp/",  # Nippon Professional Baseball
            "https://www.j-league.jp/",  # J-League
            "https://www.basketball.or.jp/",  # Japan Basketball Association
            "https://www.jva.or.jp/",  # Japan Volleyball Association
            "https://www.jta-tennis.or.jp/",  # Japan Tennis Association
        ]
        
        collected_logos = []
        
        for website in websites:
            try:
                logger.info(f"Collecting from {website}")
                logos = self._scrape_website_logos(website)
                collected_logos.extend(logos)
                time.sleep(2)  # Be respectful to servers
            except Exception as e:
                logger.error(f"Error collecting from {website}: {e}")
                
        return collected_logos
    
    def _scrape_website_logos(self, url: str) -> List[Dict]:
        """Scrape logos from a specific website"""
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Find all img tags
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")
            
            logos = []
            for img in img_elements:
                try:
                    src = img.get_attribute("src")
                    if src and self._is_logo_candidate(src, img):
                        logo_data = {
                            "url": src,
                            "alt_text": img.get_attribute("alt") or "",
                            "width": img.get_attribute("width"),
                            "height": img.get_attribute("height"),
                            "source_website": url
                        }
                        logos.append(logo_data)
                except Exception as e:
                    continue
                    
            return logos
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    def _is_logo_candidate(self, src: str, img_element) -> bool:
        """Determine if an image is likely a logo"""
        # Check file extension
        if not any(src.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            return False
            
        # Check if it's a data URL (skip)
        if src.startswith('data:'):
            return False
            
        # Check size attributes
        try:
            width = img_element.get_attribute("width")
            height = img_element.get_attribute("height")
            
            if width and height:
                w, h = int(width), int(height)
                min_w, min_h = LOGO_CHARACTERISTICS["min_size"]
                max_w, max_h = LOGO_CHARACTERISTICS["max_size"]
                
                if min_w <= w <= max_w and min_h <= h <= max_h:
                    return True
        except:
            pass
            
        # Check alt text for logo-related keywords
        alt_text = (img_element.get_attribute("alt") or "").lower()
        logo_keywords = ["logo", "sponsor", "sponsor", "パートナー", "ロゴ", "スポンサー"]
        
        return any(keyword in alt_text for keyword in logo_keywords)
    
    def download_logos(self, logo_data: List[Dict]) -> List[str]:
        """Download logo images and save them locally"""
        downloaded_paths = []
        
        for i, logo in enumerate(logo_data):
            try:
                # Convert relative URLs to absolute
                if logo["url"].startswith("//"):
                    logo["url"] = "https:" + logo["url"]
                elif logo["url"].startswith("/"):
                    logo["url"] = "https://" + logo["source_website"].split("/")[2] + logo["url"]
                
                # Download image
                response = self.session.get(logo["url"], timeout=10)
                response.raise_for_status()
                
                # Determine file extension
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                else:
                    ext = '.jpg'  # Default
                
                # Save image
                filename = f"logo_{i:04d}{ext}"
                filepath = RAW_DATA_DIR / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Validate image
                if self._validate_image(filepath):
                    downloaded_paths.append(str(filepath))
                    logger.info(f"Downloaded: {filename}")
                else:
                    filepath.unlink()  # Delete invalid image
                    
            except Exception as e:
                logger.error(f"Error downloading logo {i}: {e}")
                continue
                
        return downloaded_paths
    
    def _validate_image(self, filepath: Path) -> bool:
        """Validate downloaded image"""
        try:
            # Try to open with PIL
            with Image.open(filepath) as img:
                # Check if image is valid
                img.verify()
                
            # Try to open with OpenCV
            img = cv2.imread(str(filepath))
            if img is None:
                return False
                
            # Check dimensions
            h, w = img.shape[:2]
            min_w, min_h = LOGO_CHARACTERISTICS["min_size"]
            max_w, max_h = LOGO_CHARACTERISTICS["max_size"]
            
            return min_w <= w <= max_w and min_h <= h <= max_h
            
        except Exception:
            return False
    
    def create_synthetic_logos(self, num_logos: int = 100) -> List[str]:
        """Create synthetic logos for data augmentation"""
        synthetic_paths = []
        
        for i in range(num_logos):
            try:
                # Create a simple synthetic logo
                logo = self._generate_synthetic_logo()
                
                filename = f"synthetic_logo_{i:04d}.png"
                filepath = RAW_DATA_DIR / filename
                
                cv2.imwrite(str(filepath), logo)
                synthetic_paths.append(str(filepath))
                
            except Exception as e:
                logger.error(f"Error creating synthetic logo {i}: {e}")
                
        return synthetic_paths
    
    def _generate_synthetic_logo(self) -> np.ndarray:
        """Generate a synthetic logo for training"""
        # Random size within logo characteristics
        min_w, min_h = LOGO_CHARACTERISTICS["min_size"]
        max_w, max_h = LOGO_CHARACTERISTICS["max_size"]
        
        width = np.random.randint(min_w, max_w + 1)
        height = np.random.randint(min_h, max_h + 1)
        
        # Create random colored background
        logo = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some geometric shapes
        center = (width // 2, height // 2)
        
        # Draw circle
        cv2.circle(logo, center, min(width, height) // 4, 
                  (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), -1)
        
        # Draw rectangle
        cv2.rectangle(logo, 
                     (center[0] - width//8, center[1] - height//8),
                     (center[0] + width//8, center[1] + height//8),
                     (255, 255, 255), 2)
        
        return logo
    
    def close(self):
        """Close selenium driver"""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    """Main function to collect logo data"""
    collector = JapaneseSportLogoCollector()
    
    try:
        # Collect from websites
        logger.info("Starting logo collection from websites...")
        logo_data = collector.collect_from_websites()
        logger.info(f"Found {len(logo_data)} logo candidates")
        
        # Download logos
        logger.info("Downloading logos...")
        downloaded_paths = collector.download_logos(logo_data)
        logger.info(f"Successfully downloaded {len(downloaded_paths)} logos")
        
        # Create synthetic logos
        logger.info("Creating synthetic logos...")
        synthetic_paths = collector.create_synthetic_logos(50)
        logger.info(f"Created {len(synthetic_paths)} synthetic logos")
        
        # Save metadata
        metadata = {
            "total_logos": len(downloaded_paths) + len(synthetic_paths),
            "real_logos": len(downloaded_paths),
            "synthetic_logos": len(synthetic_paths),
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(RAW_DATA_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Logo collection completed successfully!")
        
    finally:
        collector.close()

if __name__ == "__main__":
    main()
