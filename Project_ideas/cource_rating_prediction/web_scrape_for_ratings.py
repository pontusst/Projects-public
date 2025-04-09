
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

'''
- First part of code is just webscaping for player ratings and score over/under par at course 
- Second part of code should be webscraping to get information about courses like hole length
    par, and if out-of-bounds are present
- Third part is linear regression to get labels for course rating 
- Fourth part is multivariate linear regression to see predict cource rating based on hole length, 
    par, and out of bounds. 
'''

edge_options = Options()
#edge_options.add_argument("--headless")
service = Service('C:/Users/pontu/Desktop/PROJEKT/edgedriver_win64/msedgedriver.exe')
driver = webdriver.Edge(service=service, options=edge_options)

try:
# Load the page
    driver.get("https://tjing.se/event/78603ea4-316a-4e00-946f-8136cf69b19e/round/ae08aae5-0d16-40f2-9ece-a62d4f3e0578/results")
    #driver.get("https://tjing.se/event/78603ea4-316a-4e00-946f-8136cf69b19e/round/ae08aae5-0d16-40f2-9ece-a62d4f3e0578/results")
    time.sleep(5)

    #time.sleep(5)  # Give time for the page to load
    page_source = driver.page_source
    if 'players' in page_source:
        print('found')

    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #time.sleep(3)  # Allow content to load
    
    #players = driver.find_elements(By.CLASS_NAME, "player")
    #print(f"🔍 Found {len(players)} players using CLASS_NAME!")

    # Extract data (example: getting event names if they're inside a div with class 'event-name')
    players = driver.find_elements(By.XPATH, "//div[contains(@class, 'player')]")
    #players = driver.find_elements(By.CLASS_NAME, "player")
    print(f"Found {len(players)} players using XPath!")
    players[0].click()
    #driver.execute_script("arguments[0].click();", players[0])
    time.sleep(2)
    #pdga_info = players[0].find_element(By.CLASS_NAME, "pdga").text 
    #pdga = driver.find_element(By.CLASS_NAME, "player")

    pdga_element = WebDriverWait(players[0], 5).until(
        EC.presence_of_element_located((By.CLASS_NAME, "pdga"))
       )
    print("RAW HTML:", pdga_element.get_attribute("outerHTML"))
    rating_text = driver.execute_script("return arguments[0].innerText;", pdga_element)

    print("Full PDGA text:", rating_text)

    # Extract only the rating number using regex
    import re
    match = re.search(r"Rating:\s*(\d+)", rating_text)

    if match:
        rating = match.group(1)
        print("Extracted Rating:", rating)
    else:
        print("❌ Rating not found!")
    #pdga_element = WebDriverWait(players[0], 5).until(
    #        EC.presence_of_element_located((By.XPATH, ".//p[contains(text(), 'Rating:')]/span[last()]")))
    span_elements = pdga_element.find_elements(By.TAG_NAME, "span")

    # Extract the last <span> element (which contains the rating)
    if len(span_elements) >= 2:
        rating = span_elements[-1].text.strip()
        print(f"🎯 First Player Rating: {rating}")
    else:
        print("❌ Rating not found!")
    #rating = pdga_info.split("Rating:")[-1].strip() if "Rating:" in pdga_info else "N/A"
    #rating_element = driver.find_element(By.XPATH, ".//p[contains(text(), 'Rating:')]/span[last()]")
    #rating = rating_element.text.strip()

    print("Player Rating:", rating)
    '''
    for player in players:
        try:
            driver.execute_script("arguments[0].click();", player)
            time.sleep(1)  # Small delay to allow expansion
            #player.click()
            wait = WebDriverWait(driver, 5)
            expanded_section = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "player.no-border"))
            )
            #rating_element = wait.until(EC.visibility_of_element_located((By.XPATH, "//p[contains(text(), 'Rating:')]/following-sibling::span[1]")))  # Second span contains the rating
            #print("HTML of rating element:", rating_element.get_attribute('outerHTML'))
            rating_element = expanded_section.find_element(By.XPATH, ".//p[contains(text(), 'Rating:')]/span[last()]")
            rating = rating_element.text.strip()

            print("Player Rating:", rating)
        except Exception as e:
            print(f"Could not get rating for a player: {str(e)}")
        '''

    players = []
    ratings = []
    scores = []
    #head = soup.find_all('title')
    #table = soup.find_all("div", class_='player')  # Replace "some_class" with actual class name

    for row in table.find_all("tr")[1:]:  # Skip header row
        cols = row.find_all("td")
        player_name = cols[0].text.strip()
        rating = cols[1].text.strip()
        score = cols[2].text.strip()

        players.append(player_name)
        ratings.append(rating)
        scores.append(score)

    df = pd.DataFrame({"Player": players, "Rating": ratings, "Score": scores})
    print(df)

    # Step 5: Save to CSV
    df.to_csv("disc_golf_data.csv", index=False)
except Exception as e:
    print(f'error -> {e}')
finally:
    driver.quit()

'''
Where i left off:
trying to find how to scrape ratings from tjing webpage. I have managed
to get a certain HTML sciript but not the ratings. When i have the ratings i need to
find the scores for that hole an that player. This will be my labeled data.
Then i need different metrics like hole length as features into model. 
'''

