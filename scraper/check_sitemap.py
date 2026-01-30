import requests

base_url = "https://www.psgbiz.com"
sitemaps = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/wp-sitemap.xml",
    "/page-sitemap.xml",
    "/post-sitemap.xml"
]

for sm in sitemaps:
    url = base_url + sm
    try:
        r = requests.head(url, timeout=5)
        print(f"{url}: {r.status_code}")
        if r.status_code == 200:
            print(f"FOUND: {url}")
    except Exception as e:
        print(f"{url}: Error {e}")
