import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import re

# Configuration
BASE_URL = "https://www.psgbiz.com/"
SITEMAP_URL = "https://www.psgbiz.com/wp-sitemap.xml"
OUTPUT_FILE = "psgbiz_data.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def is_internal(url):
    """Check if the URL is internal to the domain."""
    domain = urlparse(BASE_URL).netloc
    return urlparse(url).netloc == domain

def normalize_url(url):
    """Normalize URL by stripping fragments."""
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path

def clean_text(text):
    """Clean extra whitespace from text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def get_sitemap_urls(sitemap_url):
    """Recursively fetch URLs from sitemaps."""
    urls = set()
    try:
        print(f"Fetching sitemap: {sitemap_url}")
        response = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch sitemap {sitemap_url}: {response.status_code}")
            return urls
        
        soup = BeautifulSoup(response.content, 'xml')
        
        # Check for sub-sitemaps
        sitemaps = soup.find_all('sitemap')
        if sitemaps:
            for sm in sitemaps:
                loc = sm.find('loc')
                if loc:
                    urls.update(get_sitemap_urls(loc.text))
        else:
            # Check for urls
            url_tags = soup.find_all('url')
            for url_tag in url_tags:
                loc = url_tag.find('loc')
                if loc:
                    urls.add(loc.text)
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
    
    return urls

def scrape_site():
    # Step 1: Get URLs from sitemap
    sitemap_urls = get_sitemap_urls(SITEMAP_URL)
    print(f"Found {len(sitemap_urls)} URLs from sitemap.")
    
    # Step 2: Initialize queue with sitemap URLs + Base URL
    queue = list(sitemap_urls)
    if BASE_URL not in sitemap_urls:
        queue.append(BASE_URL)
        
    visited = set()
    data = []

    print(f"Starting crawl of {len(queue)} pages...")

    while queue:
        url = queue.pop(0)
        normalized_url = normalize_url(url)

        if normalized_url in visited:
            continue
        
        visited.add(normalized_url)
        print(f"Scraping ({len(visited)}/{len(sitemap_urls)+1}): {normalized_url}")

        try:
            response = requests.get(normalized_url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if response.status_code != 200:
                print(f"Failed to retrieve {normalized_url}: Status {response.status_code}")
                continue

            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                print(f"Skipping non-HTML content: {content_type}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract Data
            title = soup.title.get_text() if soup.title else ""
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "noscript", "iframe"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text(separator=' ')
            cleaned_content = clean_text(text_content)

            # Structure the data
            page_data = {
                "url": normalized_url,
                "title": clean_text(title),
                "content": cleaned_content,
                "metadata": {
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            data.append(page_data)

            # Optional: Still crawl links if we want to find pages NOT in sitemap, 
            # but for now rely on sitemap for the bulk and link crawling for deep discovery if needed.
            # To be safe, let's ADD links found to the queue if not visited.
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Filter links
                if is_internal(full_url) and full_url.startswith("http"):
                    norm_link = normalize_url(full_url)
                    
                    # Extension filter
                    if any(norm_link.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.css', '.js', '.xml', '.ico', '.svg', '.mp4']):
                        continue

                    if norm_link not in visited and norm_link not in queue and norm_link not in sitemap_urls:
                         # Only add if not already known from sitemap (to avoid re-adding)
                         # Actually if it's in sitemap it's already in queue (or popped).
                         # We check if it was ever in sitemap_urls or visited.
                         if norm_link not in visited:
                             queue.append(norm_link)

            # Be polite
            time.sleep(0.5)
            
            # Save periodically
            if len(data) % 5 == 0:
                print(f"Saving {len(data)} pages...")
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error scraping {normalized_url}: {e}")

    # Final Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Scraping complete. Found {len(data)} pages. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_site()
