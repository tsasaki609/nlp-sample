import requests

def get_bing_suggest(query):
    url = f"https://api.bing.com/osjson.aspx?query={query}"
    headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    suggestions = response.json()[1]
    return suggestions

query = "hoo"
suggestions = get_bing_suggest(query)
print(suggestions)