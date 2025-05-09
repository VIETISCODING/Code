from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os
import uuid
import pandas as pd
import re
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Tiêu đề trình duyệt để mô phỏng người dùng thực
browser_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Cài đặt đầu ra
output_folder = '.'
output_file = 'results.csv'

# Cấu hình cho các bảng dữ liệu cần cào
data_tables = {
    'general': {
        'endpoint': 'stats',
        'fields': ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls.1', 'Ast.1', 'xG.1', 'xAG.1'],
        'field_map': {
            'Nation': 'Nation', 'Pos': 'Position', 'Squad': 'Team', 'Age': 'Age', 'MP': 'Matches Played', 'Starts': 'Starts', 'Min': 'Minutes',
            'Gls': 'Goals', 'Ast': 'Assists', 'CrdY': 'Yellow Cards', 'CrdR': 'Red Cards', 'xG': 'xG', 'npxG': 'npxG', 'xAG': 'xAG',
            'PrgC': 'PrgC', 'PrgP': 'PrgP', 'PrgR': 'PrgR', 'Gls.1': 'Gls/90', 'Ast.1': 'Ast/90', 'xG.1': 'xG/90', 'xAG.1': 'xAG/90'
        }
    },
    'goalkeeping': {
        'endpoint': 'keepers',
        'fields': ['Player', 'GA90', 'Save%', 'CS%', 'Save%.1'],
        'field_map': {'GA90': 'GA90', 'Save%': 'Save%', 'CS%': 'CS%', 'Save%.1': 'PK Save%'}
    },
    'shooting': {
        'endpoint': 'shooting',
        'fields': ['Player', 'SoT%', 'SoT/90', 'G/Sh', 'Dist'],
        'field_map': {'SoT%': 'SoT%', 'SoT/90': 'SoT/90', 'G/Sh': 'G/Sh', 'Dist': 'Dist'}
    },
    'passing': {
        'endpoint': 'passing',
        'fields': ['Player', 'Cmp', 'Cmp%', 'TotDist', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP'],
        'field_map': {
            'Cmp': 'Total Cmp', 'Cmp%': 'Total Cmp%', 'TotDist': 'Total Pass Dist',
            'Cmp%.1': 'Short Cmp%', 'Cmp%.2': 'Medium Cmp%', 'Cmp%.3': 'Long Cmp%',
            'KP': 'Key Passes', '1/3': 'Pass Final 1/3', 'PPA': 'PPA', 'CrsPA': 'CrsPA', 'PrgP': 'Prog Passes'
        }
    },
    'goal_creation': {
        'endpoint': 'gca',
        'fields': ['Player', 'SCA', 'SCA90', 'GCA', 'GCA90'],
        'field_map': {'SCA': 'SCA', 'SCA90': 'SCA90', 'GCA': 'GCA', 'GCA90': 'GCA90'}
    },
    'defensive': {
        'endpoint': 'defense',
        'fields': ['Player', 'Tkl', 'TklW', 'Att', 'Lost', 'Blocks', 'Sh', 'Pass', 'Int'],
        'field_map': {
            'Tkl': 'Tackles', 'TklW': 'Tackles Won', 'Att': 'Challenges Att', 'Lost': 'Challenges Lost',
            'Blocks': 'Blocks', 'Sh': 'Shots Blocked', 'Pass': 'Passes Blocked', 'Int': 'Interceptions'
        }
    },
    'possession': {
        'endpoint': 'possession',
        'fields': ['Player', 'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att', 'Succ%', 'Tkld%', 'Carries', 'TotDist', 'PrgDist', 'PrgC', '1/3', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR'],
        'field_map': {
            'Touches': 'Touches', 'Def Pen': 'Def Pen Touches', 'Def 3rd': 'Def 3rd Touches', 'Mid 3rd': 'Mid 3rd Touches', 'Att 3rd': 'Att 3rd Touches', 'Att Pen': 'Att Pen Touches',
            'Att': 'Take-Ons Att', 'Succ%': 'Take-On Succ%', 'Tkld%': 'Take-On Tkld%',
            'Carries': 'Carries', 'TotDist': 'Carry Dist', 'PrgDist': 'Prog Carry Dist',
            'PrgC': 'Prog Carries', '1/3': 'Carry Final 1/3', 'CPA': 'Carry Pen Area', 'Mis': 'Miscontrols', 'Dis': 'Dispossessed',
            'Rec': 'Passes Received', 'PrgR': 'Prog Passes Received'
        }
    },
    'miscellaneous': {
        'endpoint': 'misc',
        'fields': ['Player', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost', 'Won%'],
        'field_map': {'Fls': 'Fouls Committed', 'Fld': 'Fouls Drawn', 'Off': 'Offsides', 'Crs': 'Crosses', 'Recov': 'Recoveries', 'Won': 'Aerials Won', 'Lost': 'Aerials Lost', 'Won%': 'Aerial Won%'}
    }
}

def fetch_table_data(table_config):
    """Lấy một bảng cụ thể từ FBref bằng Selenium và xử lý thành DataFrame."""
    table_url = f'https://fbref.com/en/comps/9/{table_config["endpoint"]}/Premier-League-Stats'
    
    # Cấu hình Selenium cho Chrome không giao diện
    chrome_opts = Options()
    chrome_opts.add_argument('--headless')
    chrome_opts.add_argument(f'user-agent={browser_headers["User-Agent"]}')
    chrome_opts.add_argument('--no-sandbox')
    chrome_opts.add_argument('--disable-dev-shm-usage')
    chrome_opts.add_argument('--disable-notifications')
    chrome_opts.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(options=chrome_opts)
    
    try:
        driver.get(table_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'stats_table'))
        )
        
        # Xác định vị trí bảng thống kê
        stats_table = None
        try:
            stats_table = driver.find_element(By.ID, f'stats_{table_config["endpoint"]}_9')
        except:
            tables = driver.find_elements(By.CLASS_NAME, 'stats_table')
            for tbl in tables:
                if tbl.find_elements(By.CSS_SELECTOR, 'td[data-stat="player"]'):
                    stats_table = tbl
                    break
        
        if not stats_table:
            print(f"Không tìm thấy bảng thống kê tại {table_url}")
            return None
        
        # Phân tích HTML của bảng bằng BeautifulSoup
        soup = BeautifulSoup(stats_table.get_attribute('outerHTML'), 'html.parser')
        
        # Trích xuất ID và tên cầu thủ
        player_ids = []
        player_names = []
        for td in soup.find_all('td', {'data-stat': 'player'}):
            link = td.find('a')
            if link and 'href' in link.attrs:
                href = link['href']
                match = re.search(r'/players/(\w+)/', href)
                if match:
                    player_ids.append(match.group(1))
                    player_names.append(td.get_text(strip=True))
                else:
                    player_ids.append(None)
                    player_names.append(None)
            else:
                player_ids.append(None)
                player_names.append(None)
        
        # Chuyển bảng thành DataFrame
        try:
            html_content = str(soup)
            df = pd.read_html(StringIO(html_content), header=[0,1])[0]
        except ValueError as e:
            print(f"Không thể phân tích bảng tại {table_url}: {e}")
            return None
        
        # Làm phẳng tên cột
        df.columns = [col[1] for col in df.columns]
        print(f"Các cột thô từ {table_url}: {df.columns.tolist()}")
        
        # Xử lý các cột trùng lặp
        ambiguous_cols = ['Gls', 'Ast', 'xG', 'xAG', 'Save%', 'Cmp', 'Cmp%', 'Tkl', 'PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def']
        renamed_cols = []
        col_counter = {}
        for col in df.columns:
            if col in ambiguous_cols:
                col_counter[col] = col_counter.get(col, 0) + 1
                if col == 'Save%':
                    renamed_cols.append('Save%_overall' if col_counter[col] == 1 else 'Save%_pk')
                elif col == 'Cmp':
                    mappings = {1: 'Cmp_total', 2: 'Cmp_short', 3: 'Cmp_medium', 4: 'Cmp_long'}
                    renamed_cols.append(mappings.get(col_counter[col], f'Cmp_{col_counter[col]}'))
                elif col == 'Cmp%':
                    mappings = {1: 'Cmp%_total', 2: 'Cmp%_short', 3: 'Cmp%_medium', 4: 'Cmp%_long'}
                    renamed_cols.append(mappings.get(col_counter[col], f'Cmp%_{col_counter[col]}'))
                elif col in ['PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def']:
                    renamed_cols.append(f'{col}_sca' if col_counter[col] == 1 else f'{col}_gca')
                else:
                    renamed_cols.append(f'{col}_total' if col_counter[col] == 1 else f'{col}_per90')
            else:
                renamed_cols.append(col)
        
        df.columns = renamed_cols
        print(f"Các cột sau khi xử lý trùng lặp: {df.columns.tolist()}")
        
        # Ánh xạ cột sang các trường mong đợi
        col_mapping = {}
        for field in table_config['fields']:
            if field == 'Save%':
                mapped_col = 'Save%_overall'
            elif field == 'Save%.1':
                mapped_col = 'Save%_pk'
            elif field == 'Cmp':
                mapped_col = 'Cmp_total'
            elif field == 'Cmp%':
                mapped_col = 'Cmp%_total'
            elif field == 'Cmp%.1':
                mapped_col = 'Cmp%_short'
            elif field == 'Cmp%.2':
                mapped_col = 'Cmp%_medium'
            elif field == 'Cmp%.3':
                mapped_col = 'Cmp%_long'
            elif field == 'Tkl':
                mapped_col = 'Tkl_total'
            elif field == 'Sh':
                mapped_col = 'Sh_sca'
            elif field == 'Fld':
                mapped_col = 'Fld_sca'
            elif field.endswith('.1'):
                base = field[:-2]
                mapped_col = f'{base}_per90'
            else:
                mapped_col = f'{field}_total' if field in ['Gls', 'Ast', 'xG', 'xAG'] else field
            
            if mapped_col in df.columns:
                col_mapping[mapped_col] = field
            else:
                col_mapping[field] = field
        
        print(f"Ánh xạ cột cho {table_url}: {col_mapping}")
        
        # Áp dụng đổi tên cột
        df = df.rename(columns=col_mapping)
        print(f"Các cột sau khi đổi tên: {df.columns.tolist()}")
        
        # Loại bỏ các cột trùng lặp
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            print(f"Tìm thấy các cột trùng lặp: {duplicates}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print(f"Các cột sau khi loại bỏ trùng lặp: {df.columns.tolist()}")
        
        # Đảm bảo cột Player tồn tại
        player_col = 'Player' if 'Player' in df.columns else next((col for col in df.columns if 'Player' in col), None)
        if not player_col:
            print(f"Không tìm thấy cột Player tại {table_url}")
            return None
        if player_col != 'Player':
            df = df.rename(columns={player_col: 'Player'})
        
        # Tạo DataFrame cho ID cầu thủ
        id_df = pd.DataFrame({'Player_ID': player_ids, 'Player': player_names})
        id_df = id_df[id_df['Player_ID'].notna() & id_df['Player'].notna()]
        
        # Lọc DataFrame cho các cầu thủ hợp lệ
        df = df[df['Player'].isin(id_df['Player'])]
        df = df.reset_index(drop=True)
        
        # Thêm Player_ID
        if len(id_df) == len(df):
            df['Player_ID'] = id_df['Player_ID'].reset_index(drop=True)
        else:
            print(f"Số lượng cầu thủ không khớp tại {table_url}: {len(id_df)} ID, {len(df)} hàng")
            return None
        
        # Chọn các cột mong muốn
        selected_cols = ['Player_ID', 'Player'] + table_config['fields'][1:] if table_config['endpoint'] == 'stats' else ['Player_ID'] + table_config['fields']
        selected_cols = list(dict.fromkeys(selected_cols))
        
        # Xử lý các cột bị thiếu
        missing = [col for col in selected_cols if col not in df.columns]
        if missing:
            print(f"Các cột bị thiếu tại {table_url}: {missing}")
            for col in missing:
                df[col] = 'N/a'
        
        df = df[selected_cols]
        
        # Áp dụng tên cột cuối cùng
        df = df.rename(columns=table_config['field_map'])
        
        # Xử lý cột Age
        if 'Age' in df.columns:
            df['Age'] = df['Age'].apply(lambda x: str(x).split('-')[0] if isinstance(x, str) and '-' in x else x)
        
        # Chuyển đổi các cột số
        non_numeric = ['Player_ID', 'Player', 'Nation', 'Position', 'Team']
        numeric_cols = [col for col in df.columns if col not in non_numeric and col in df.columns]
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Không thể chuyển đổi {col} thành số: {e}")
                    df[col] = 'N/a'
            else:
                print(f"Cột {col} bị thiếu trong DataFrame")
                df[col] = 'N/a'
        
        print(f"Các cột cuối cùng: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ {table_url}: {e}")
        return None
    finally:
        driver.quit()

# Thu thập dữ liệu từ tất cả các bảng
table_data = {}
failed_fetches = []
for table_name, config in data_tables.items():
    result = fetch_table_data(config)
    if result is not None:
        table_data[table_name] = result
    else:
        failed_fetches.append(table_name)
        print(f"Không thể lấy bảng {table_name}, tiếp tục...")

if not table_data:
    print("Không lấy được dữ liệu nào thành công. Thoát.")
    exit(1)

# Lọc các cầu thủ có thời gian thi đấu >90 phút
if 'general' in table_data:
    base_df = table_data['general']
    base_df['Minutes'] = pd.to_numeric(base_df['Minutes'], errors='coerce')
else:
    print("Bảng thống kê chung bị thiếu, sử dụng bảng có sẵn đầu tiên")
    base_df = None

# Gộp tất cả các bảng
if base_df is not None:
    combined_df = base_df
else:
    combined_df = next(iter(table_data.values()), None)
    if combined_df is None:
        print("Không có dữ liệu để gộp. Thoát.")
        exit(1)

for table_name in data_tables:
    if table_name != 'general' and table_name in table_data:
        merge_cols = ['Player_ID'] + list(data_tables[table_name]['field_map'].values())
        merge_cols = list(dict.fromkeys(merge_cols))
        available = [col for col in merge_cols if col in table_data[table_name].columns]
        if len(available) < len(merge_cols):
            print(f"Cảnh báo: Thiếu cột trong {table_name} để gộp: {[col for col in merge_cols if col not in available]}")
        combined_df = pd.merge(combined_df, table_data[table_name][available], on='Player_ID', how='left')

# Tổng hợp dữ liệu theo cầu thủ
aggregation_rules = {
    'Player': 'first', 'Nation': 'first', 'Position': 'first', 'Team': lambda x: ', '.join(x.dropna().unique()),
    'Age': 'first', 'Matches Played': 'sum', 'Starts': 'sum', 'Minutes': 'sum', 'Goals': 'sum', 'Assists': 'sum',
    'Yellow Cards': 'sum', 'Red Cards': 'sum', 'xG': 'sum', 'npxG': 'sum', 'xAG': 'sum',
    'PrgC': 'sum', 'PrgP': 'sum', 'PrgR': 'sum', 'Gls/90': 'mean', 'Ast/90': 'mean', 'xG/90': 'mean', 'xAG/90': 'mean',
    'GA90': 'mean', 'Save%': 'mean', 'CS%': 'mean', 'PK Save%': 'mean',
    'SoT%': 'mean', 'SoT/90': 'mean', 'G/Sh': 'mean', 'Dist': 'mean',
    'Total Cmp': 'sum', 'Total Cmp%': 'mean', 'Total Pass Dist': 'sum',
    'Short Cmp%': 'mean', 'Medium Cmp%': 'mean', 'Long Cmp%': 'mean',
    'Key Passes': 'sum', 'Pass Final 1/3': 'sum', 'PPA': 'sum', 'CrsPA': 'sum', 'Prog Passes': 'sum',
    'SCA': 'sum', 'SCA90': 'mean', 'GCA': 'sum', 'GCA90': 'mean',
    'Tackles': 'sum', 'Tackles Won': 'sum', 'Challenges Att': 'sum', 'Challenges Lost': 'sum',
    'Blocks': 'sum', 'Shots Blocked': 'sum', 'Passes Blocked': 'sum', 'Interceptions': 'sum',
    'Touches': 'sum', 'Def Pen Touches': 'sum', 'Def 3rd Touches': 'sum', 'Mid 3rd Touches': 'sum', 'Att 3rd Touches': 'sum', 'Att Pen Touches': 'sum',
    'Take-Ons Att': 'sum', 'Take-On Succ%': 'mean', 'Take-On Tkld%': 'mean',
    'Carries': 'sum', 'Carry Dist': 'sum', 'Prog Carry Dist': 'sum',
    'Prog Carries': 'sum', 'Carry Final 1/3': 'sum', 'Carry Pen Area': 'sum', 'Miscontrols': 'sum', 'Dispossessed': 'sum',
    'Passes Received': 'sum', 'Prog Passes Received': 'sum',
    'Fouls Committed': 'sum', 'Fouls Drawn': 'sum', 'Offsides': 'sum', 'Crosses': 'sum', 'Recoveries': 'sum',
    'Aerials Won': 'sum', 'Aerials Lost': 'sum', 'Aerial Won%': 'mean'
}

combined_df = combined_df.groupby('Player_ID').agg(aggregation_rules).reset_index(drop=True)
combined_df = combined_df[combined_df['Minutes'] > 90]

# Sắp xếp theo tên đầu tiên
combined_df['First_Name'] = combined_df['Player'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown')
combined_df = combined_df.sort_values(by='First_Name')
combined_df = combined_df.drop(columns=['First_Name'])

# Xử lý giá trị bị thiếu
combined_df = combined_df.fillna("N/a")

# Xác định thứ tự cột đầu ra
output_columns = [
    'Player', 'Nation', 'Team', 'Position', 'Age', 'Matches Played', 'Starts', 'Minutes', 'Goals', 'Assists', 'Yellow Cards', 'Red Cards',
    'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls/90', 'Ast/90', 'xG/90', 'xAG/90',
    'GA90', 'Save%', 'CS%', 'PK Save%', 'SoT%', 'SoT/90', 'G/Sh', 'Dist',
    'Total Cmp', 'Total Cmp%', 'Total Pass Dist', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'Key Passes', 'Pass Final 1/3', 'PPA', 'CrsPA', 'Prog Passes',
    'SCA', 'SCA90', 'GCA', 'GCA90', 'Tackles', 'Tackles Won', 'Challenges Att', 'Challenges Lost', 'Blocks', 'Shots Blocked', 'Passes Blocked', 'Interceptions',
    'Touches', 'Def Pen Touches', 'Def 3rd Touches', 'Mid 3rd Touches', 'Att 3rd Touches', 'Att Pen Touches', 'Take-Ons Att', 'Take-On Succ%', 'Take-On Tkld%',
    'Carries', 'Carry Dist', 'Prog Carry Dist', 'Prog Carries', 'Carry Final 1/3', 'Carry Pen Area', 'Miscontrols', 'Dispossessed',
    'Passes Received', 'Prog Passes Received', 'Fouls Committed', 'Fouls Drawn', 'Offsides', 'Crosses', 'Recoveries', 'Aerials Won', 'Aerials Lost', 'Aerial Won%'
]

# Đảm bảo tất cả các cột tồn tại
for col in output_columns:
    if col not in combined_df.columns:
        combined_df[col] = "N/a"

final_df = combined_df[output_columns]

# Làm sạch cột Nation
def clean_nation(nation):
    if pd.isna(nation) or not isinstance(nation, str) or not nation.strip():
        return "N/a"
    return ''.join(char for char in nation if char.isupper()) or "N/a"

final_df['Nation'] = final_df['Nation'].apply(clean_nation)

# Kiểm tra chất lượng dữ liệu
for col in output_columns:
    na_count = final_df[col].eq("N/a").sum()
    if na_count > len(final_df) * 0.5:
        print(f"Cảnh báo: Cột {col} có {na_count} giá trị 'N/a' ({na_count/len(final_df)*100:.1f}% số hàng)")

# Lưu vào CSV
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, output_file)
final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"Dữ liệu đã được lưu vào {csv_path}")
if failed_fetches:
    print(f"Cảnh báo: Không thể lấy các bảng: {failed_fetches}")