import requests

# 1. Get backend images (so you can use their data URIs)
backend_url = "http://localhost:8008/list-backend-images"
resp = requests.get(backend_url)
images = resp.json()

# 2. Filter for a specific country (e.g., Germany)
country_name = "Germany"
country_code = "DE"
germany_images = [img for img in images if img.get("countryName") == country_name]

# Use up to 10 images for the test
image_urls = [img["dataUri"] for img in germany_images[:10]]

if not image_urls:
    print("No images found for", country_name)
    exit(1)

# 3. Run clustering with psychology enabled
clustering_payload = {
    "imageUrls": image_urls,
    "gridSize": 5,
    "clusteringStrategy": "enhanced_features",
    "colorSpace": "lab",
    "numberOfColors": 5,
    "clusteringMode": "som",
    "includePsychology": True,
    "culturalContext": "universal"
}

clustering_url = "http://localhost:8008/cluster-postcard-images"
clustering_resp = requests.post(clustering_url, json=clustering_payload)
print("Clustering status:", clustering_resp.status_code)
print("Clustering response (truncated):", str(clustering_resp.json())[:500], "...")

# 4. Test the individual country psychology endpoint
psychology_url = f"http://localhost:8008/countries/{country_code}/psychology"
params = {"cultural_context": "universal"}
psychology_resp = requests.get(psychology_url, params=params)
print("\nIndividual country psychology status:", psychology_resp.status_code)
print("Individual country psychology response:", psychology_resp.json())