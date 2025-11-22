from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import requests
from textblob import TextBlob
import pandas as pd


def scrape_google_reviews(business_name, location):
    """
    Scrapes Google reviews for the given business name and location.
    """
    url = f"https://www.google.com/search?q={business_name}+{location}+reviews"
    driver = webdriver.Chrome()  # Adjust this if you're using another browser
    driver.get(url)
    time.sleep(3)  # Allow the page to load

    # Extract reviews
    reviews = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    review_elements = soup.find_all('span', class_='review-full-text')  # Adjust based on the actual class name
    for element in review_elements:
        reviews.append(element.text.strip())

    driver.quit()
    return reviews


def scrape_instagram_mentions(business_name, location):
    """
    Scrapes Instagram mentions for the given business name and location.
    """
    # Construct search query
    url = f"https://www.instagram.com/explore/tags/{business_name.replace(' ', '')}/"
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)

    # Scroll down to load more posts
    for _ in range(5):  # Adjust scrolls based on how much data you want
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
        time.sleep(2)

    # Extract posts
    posts = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    post_elements = soup.find_all('div', class_='_aap6 _aap7 _aap8')  # Adjust based on the actual class name
    for element in post_elements:
        posts.append(element.text.strip())

    driver.quit()
    return posts


def analyze_sentiment(data):
    """
    Analyzes sentiment of the given data.
    """
    sentiments = []
    for text in data:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiments.append({'text': text, 'polarity': polarity, 'sentiment': sentiment})
    return sentiments


def main():
    business_name = input("Enter the business name: ")
    location = input("Enter the location: ")

    print("Scraping Google Reviews...")
    google_reviews = scrape_google_reviews(business_name, location)
    print(f"Found {len(google_reviews)} reviews on Google.")

    print("Scraping Instagram Mentions...")
    instagram_mentions = scrape_instagram_mentions(business_name, location)
    print(f"Found {len(instagram_mentions)} mentions on Instagram.")

    print("Analyzing Sentiment for Google Reviews...")
    google_sentiment = analyze_sentiment(google_reviews)

    print("Analyzing Sentiment for Instagram Mentions...")
    instagram_sentiment = analyze_sentiment(instagram_mentions)

    # Print and save results
    df_google = pd.DataFrame(google_sentiment)
    df_instagram = pd.DataFrame(instagram_sentiment)

    print("Google Reviews Sentiment:")
    print(df_google)

    print("Instagram Mentions Sentiment:")
    print(df_instagram)

    # Save to CSV
    df_google.to_csv(f"{business_name}_google_sentiment.csv", index=False)
    df_instagram.to_csv(f"{business_name}_instagram_sentiment.csv", index=False)


if __name__ == "__main__":
    main()
