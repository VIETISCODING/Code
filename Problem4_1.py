from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO
from rapidfuzz import process, fuzz
import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Set up Chrome options for running in headless mode
browser_options = Options()
browser_options.add_argument("--headless")
browser_options.add_argument("--disable-gpu")
browser_options.add_argument("--no-sandbox")

# Initialize the browser with ChromeDriverManager
browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_options)

print("Fetching data from FBref...")
# Specify the URL and table identifier for FBref
premier_league_url = "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats"
target_table_id = "stats_standard"
selected_columns = ["Player", "Nation", "Team", "Position", "Age", "Minutes"]

# Mapping for renaming DataFrame columns
column_name_map = {
    "Unnamed: 1": "Player",
    "Unnamed: 2": "Nation",
    "Unnamed: 3": "Position",
    "Unnamed: 4": "Team",
    "Unnamed: 5": "Age",
    "Playing Time.2": "Minutes"
}

# Load the FBref page and allow time for rendering
browser.get(premier_league_url)
time.sleep(3)

# Parse the page content using BeautifulSoup
html_content = BeautifulSoup(browser.page_source, "html.parser")
page_comments = html_content.find_all(string=lambda text: isinstance(text, Comment))

# Locate the desired table within comments
data_table = None
for comment in page_comments:
    if target_table_id in comment:
        parsed_comment = BeautifulSoup(comment, "html.parser")
        data_table = parsed_comment.find("table", {"id": target_table_id})
        if data_table:
            break

# Check if the table was found, raise error if not
if not data_table:
    raise Exception(f"Could not locate table {target_table_id}")

# Extract table data into a pandas DataFrame
premier_league_data = pd.read_html(StringIO(str(data_table)), header=0)[0]

# Apply column renaming and clean the DataFrame
premier_league_data = premier_league_data.rename(columns=column_name_map)
premier_league_data = premier_league_data.loc[:, ~premier_league_data.columns.duplicated()]
premier_league_data = premier_league_data[[col for col in premier_league_data.columns if col in selected_columns]]
premier_league_data = premier_league_data.drop_duplicates(subset=["Player"], keep="first")

# Convert 'Minutes' to numeric and filter for players with over 900 minutes
premier_league_data["Minutes"] = pd.to_numeric(premier_league_data["Minutes"], errors="coerce")
premier_league_data = premier_league_data[premier_league_data["Minutes"] > 900]
premier_league_data = premier_league_data[["Player", "Nation", "Team", "Position", "Age"]].fillna("N/A")

# Standardize player names for matching
premier_league_data["standardized_name"] = premier_league_data["Player"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

print("Gathering data from FootballTransfers...")
# Define the base URL and create list of paginated URLs
transfer_base_url = "https://www.footballtransfers.com/en/players/uk-premier-league"
transfer_page_urls = [transfer_base_url] + [f"{transfer_base_url}/{i}" for i in range(2, 23)]

# Collect player data from all pages
transfer_records = []
for page_url in transfer_page_urls:
    print(f"Scraping: {page_url}")
    try:
        # Access the page and wait for it to load
        browser.get(page_url)
        time.sleep(5)
        page_content = BeautifulSoup(browser.page_source, 'html.parser')

        # Find the table body with player information
        table_content = page_content.find('tbody', id='player-table-body')
        if not table_content:
            print(f"No table found on {page_url}")
            continue

        # Process each row in the table
        table_rows = table_content.find_all('tr')
        for row in table_rows:
            player_info = row.find('td', class_='td-player')
            player_full_name = None
            if player_info:
                name_container = player_info.find('div', class_='text')
                if name_container and name_container.a:
                    player_full_name = name_container.a.get('title')

            value_info = row.find('td', class_='text-center')
            value_tag = value_info.find('span', class_='player-tag') if value_info else None
            player_value = value_tag.get_text(strip=True) if value_tag else None

            if player_full_name and player_value:
                transfer_records.append({'Name': player_full_name, 'Value': player_value})

    except Exception as e:
        print(f"Error processing {page_url}: {e}")

# Shut down the browser
browser.quit()

# Create a DataFrame from the collected transfer data
transfer_data_frame = pd.DataFrame(transfer_records)
transfer_data_frame["standardized_name"] = transfer_data_frame["Name"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

# Prepare for name matching between datasets
transfer_name_list = transfer_data_frame["standardized_name"].tolist()
player_value_map = {}

# Perform fuzzy matching to assign player values
for name in premier_league_data["standardized_name"]:
    match, score, index = process.extractOne(name, transfer_name_list, scorer=fuzz.token_sort_ratio)
    if score >= 90:
        matched_player_value = transfer_data_frame.iloc[index]["Value"]
        player_value_map[name] = matched_player_value
    else:
        player_value_map[name] = "N/A"

# Add market values to the main DataFrame
premier_league_data["Value"] = premier_league_data["standardized_name"].map(player_value_map)

# Finalize the DataFrame and export to CSV
output_data = premier_league_data.drop(columns=["standardized_name"])[["Player", "Nation", "Team", "Position", "Age", "Value"]]
output_data.to_csv("players_with_values.csv", index=False, encoding="utf-8-sig")
print(f"File saved: players_with_values.csv with {output_data.shape[0]} players.")