import requests
import xml.etree.ElementTree as ET
import json
import time
import sqlite3
from datetime import datetime, timedelta

DB_NAME = "bgg_cache.db"
CACHE_EXPIRY_DAYS = 7  
RETRY_DELAY = 5  
MAX_RETRIES = 3  


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS forums (
        game_id INTEGER PRIMARY KEY, 
        forum_id INTEGER, 
        last_updated TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS threads (
        thread_id INTEGER PRIMARY KEY, 
        forum_id INTEGER, 
        subject TEXT, 
        author TEXT, 
        numarticles INTEGER, 
        postdate TEXT, 
        lastpostdate TEXT, 
        completed INTEGER DEFAULT 0, 
        last_updated TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS posts (
        post_id INTEGER PRIMARY KEY, 
        thread_id INTEGER, 
        username TEXT, 
        post_date TEXT, 
        content TEXT, 
        last_updated TEXT
    )''')

    conn.commit()
    conn.close()


def api_request(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
            else:
                print(f"API Error ({response.status_code}): {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(RETRY_DELAY)
    return None


def get_forum_id(game_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT forum_id FROM forums WHERE game_id = ?", (game_id,))
    row = cur.fetchone()
    conn.close()
    
    if row:
        return row[0]  

    response = api_request(f"https://boardgamegeek.com/xmlapi2/forumlist?id={game_id}&type=thing")
    if not response:
        return None

    root = ET.fromstring(response.content)
    for forum in root.findall("forum"):
        if forum.get("title", "").lower() == "rules":
            forum_id = forum.get("id")
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO forums (game_id, forum_id, last_updated) VALUES (?, ?, ?)",
                        (game_id, forum_id, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return forum_id

    return None


def get_threads(forum_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT thread_id FROM threads WHERE forum_id = ?", (forum_id,))
    existing_threads = {row[0] for row in cur.fetchall()}  
    conn.close()

    new_threads = []
    page = 1

    while True:
        response = api_request(f"https://boardgamegeek.com/xmlapi2/forum?id={forum_id}&page={page}&sort=hot")
        if not response:
            break

        root = ET.fromstring(response.content)
        thread_elements = root.findall(".//thread")
        
        if not thread_elements:
            break  

        for thread in thread_elements:
            thread_id = thread.get("id")
            thread_data = {
                "thread_id": thread_id,
                "forum_id": forum_id,
                "subject": thread.get("subject"),
                "author": thread.get("author"),
                "numarticles": thread.get("numarticles"),
                "postdate": thread.get("postdate"),
                "lastpostdate": thread.get("lastpostdate"),
                "completed": 0,
                "last_updated": datetime.now().isoformat(),
                "url": f"https://boardgamegeek.com/thread/{thread_id}"
            }

            new_threads.append(thread_data)

        page += 1
        time.sleep(1)  

    return new_threads


def get_posts(thread_id):
    response = api_request(f"https://boardgamegeek.com/xmlapi2/thread?id={thread_id}")
    if not response:
        return []

    root = ET.fromstring(response.content)
    new_posts = []

    for post in root.findall(".//article"):
        new_posts.append({
            "post_id": post.get("id"),
            "thread_id": thread_id,
            "username": post.get("username"),
            "post_date": post.get("postdate"),
            "content": post.find("body").text if post.find("body") is not None else "",
        })

    return new_posts


def scrape_bgg_forum(game_id):
    init_db()
    
    forum_id = get_forum_id(game_id)
    if not forum_id:
        return

    print(f"Found Rules Forum ID: {forum_id}")
    threads = get_threads(forum_id)
    print(f"Found {len(threads)} new threads.")

    grouped_data = {}

    for thread in threads:
        thread_id = thread["thread_id"]
        print(f"Fetching posts from thread {thread_id} - {thread['subject']}...")
        posts = get_posts(thread_id)

        grouped_data[thread_id] = {
            "subject": thread["subject"],
            "author": thread["author"],
            "numarticles": thread["numarticles"],
            "url": thread["url"],
            "posts": posts
        }

        time.sleep(1)

    with open("bgg_rules_posts.json", "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=4, ensure_ascii=False)

    print("Scraping complete. Data saved to bgg_rules_posts_sorted.json")

scrape_bgg_forum(308119)
