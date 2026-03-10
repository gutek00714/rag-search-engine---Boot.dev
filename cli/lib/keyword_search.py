import json

def keyword_search(query):
    with open('data/movies.json', 'r') as f:
        data = json.load(f)

    movies_list = data["movies"]
    results = []
    i=1
    for item in movies_list:
        if query in item["title"]:
            results.append(item)
            print(f"{i}. {item['title']}")
            i+=1

        if len(results) == 5:
            break