import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import warnings
import time
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

# Danh sách URL cho các bảng thống kê khác nhau
danhSachURL = [
    ('https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats#stats_standard', 'stats_standard'),
    ('https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats#stats_keeper', 'stats_keeper'),
    ('https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats#stats_shooting', 'stats_shooting'),
    ('https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats#stats_passing', 'stats_passing'),
    ('https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats#stats_gca', 'stats_gca'),
    ('https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats#stats_defense', 'stats_defense'),
    ('https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats#stats_possession', 'stats_possession'),
    ('https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats#stats_misc', 'stats_misc')
]

# Bản đồ ánh xạ các thống kê cần thiết với tên cột của fbref
bangThongKeCanThiet = {
    'Nation': 'Nation',
    'Team': 'Squad',
    'Position': 'Pos',
    'Age': 'Age',
    'Matches Played': 'MP',
    'Starts': 'Starts',
    'Minutes': 'Min',
    'Goals': 'Gls',
    'Assists': 'Ast',
    'Yellow Cards': 'CrdY',
    'Red Cards': 'CrdR',
    'xG': 'xG',
    'xAG': 'xAG',
    'PrgC': 'PrgC',
    'PrgP': 'PrgP',
    'PrgR': 'PrgR',
    'Gls/90': 'Gls_2',
    'Ast/90': 'Ast_2',
    'xG/90': 'xG_2',
    'xAG/90': 'xAG_2',
    'GA90': 'GA90',
    'Save%': 'Save%',
    'CS%': 'CS%',
    'PK Save%': 'Save%_2',
    'SoT%': 'SoT%',
    'SoT/90': 'SoT/90',
    'G/Sh': 'G/Sh',
    'Dist': 'Dist',
    'Cmp': 'Cmp',
    'Cmp%': 'Cmp%',
    'TotDist': 'TotDist',
    'Short Cmp%': 'Cmp%_2',
    'Medium Cmp%': 'Cmp%_3',
    'Long Cmp%': 'Cmp%_4',
    'KP': 'KP',
    '1/3': '1/3',
    'PPA': 'PPA',
    'CrsPA': 'CrsPA',
    'PrgP (Passing)': 'PrgP',
    'SCA': 'SCA',
    'SCA90': 'SCA90',
    'GCA': 'GCA',
    'GCA90': 'GCA90',
    'Tkl': 'Tkl',
    'TklW': 'TklW',
    'Att (Challenges)': 'Att',
    'Lost (Challenges)': 'Lost',
    'Blocks': 'Blocks',
    'Sh (Blocks)': 'Sh',
    'Pass (Blocks)': 'Pass',
    'Int': 'Int',
    'Touches': 'Touches',
    'Def Pen': 'Def Pen',
    'Def 3rd': 'Def 3rd',
    'Mid 3rd': 'Mid 3rd',
    'Att 3rd': 'Att 3rd',
    'Att Pen': 'Att Pen',
    'Att (Take-Ons)': 'Att',
    'Succ% (Take-Ons)': 'Succ%',
    'Tkld%': 'Tkld%',
    'Carries': 'Carries',
    'ProDist': 'PrgDist',
    'ProgC (Carries)': 'PrgC',
    '1/3 (Carries)': '1/3',
    'CPA': 'CPA',
    'Mis': 'Mis',
    'Dis': 'Dis',
    'Rec': 'Rec',
    'PrgR (Receiving)': 'PrgR',
    'Fls': 'Fls',
    'Fld': 'Fld',
    'Off': 'Off',
    'Crs': 'Crs',
    'Recov': 'Recov',
    'Won (Aerial)': 'Won',
    'Lost (Aerial)': 'Lost',
    'Won% (Aerial)': 'Won%'
}

def lamSachTenCauThu(ten):
    """Làm sạch tên cầu thủ bằng cách loại bỏ ký tự đặc biệt và khoảng trắng thừa."""
    return re.sub(r'[^\w\s]', '', ten.strip()) if isinstance(ten, str) else ''

def layTenDau(ten):
    """Lấy tên đầu tiên từ tên cầu thủ."""
    phan = ten.split()
    return phan[0] if phan else ten

def layDuLieuBang(url, maBang, soLanThuLai=7):
    """Lấy dữ liệu từ bảng được chỉ định từ URL với số lần thử lại."""
    print(f"Đang lấy dữ liệu từ {url} cho bảng ID {maBang}")
    tuyChon = Options()
    tuyChon.add_argument('--headless')
    tuyChon.add_argument('--disable-gpu')
    tuyChon.add_argument('--no-sandbox')
    tuyChon.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    tuyChon.add_argument('--disable-blink-features=AutomationControlled')
    tuyChon.add_argument('--ignore-certificate-errors')
    
    for lanThu in range(soLanThuLai):
        trinhDieuKhien = webdriver.Chrome(options=tuyChon)
        try:
            trinhDieuKhien.get(url)
            doi = WebDriverWait(trinhDieuKhien, 15)
            bang = doi.until(EC.presence_of_element_located((By.ID, maBang)))
            htmlBang = bang.get_attribute('outerHTML')
            
            sup = BeautifulSoup(htmlBang, 'html.parser')
            
            danhSachHang = []
            for i in range(0, 1000):
                hang = sup.find('tr', {'data-row': str(i)})
                if hang:
                    danhSachHang.append(hang)
            
            duLieu = []
            for hang in danhSachHang:
                o = hang.find_all(['td', 'th'])
                duLieuHang = [o.get_text(strip=True) if o.get_text(strip=True) else "N/a" for o in o]
                if duLieuHang and len(duLieuHang) > 1:
                    duLieu.append(duLieuHang)
            
            hangTieuDe = sup.find('tr', class_='thead')
            if not hangTieuDe:
                hangTieuDe = sup.find_all('tr')[0]
            tieuDe = [th.get_text(strip=True) if th.get_text(strip=True) else f"Col_{i}" 
                      for i, th in enumerate(hangTieuDe.find_all(['th', 'td']))]
            
            if duLieu:
                soCotToiDa = max(len(hang) for hang in duLieu)
                if len(tieuDe) > soCotToiDa:
                    tieuDe = tieuDe[:soCotToiDa]
                elif len(tieuDe) < soCotToiDa:
                    tieuDe = tieuDe + [f'Col_{i}' for i in range(len(tieuDe), soCotToiDa)]
                
                khungDuLieu = pd.DataFrame(duLieu, columns=tieuDe)
                
                demCot = {}
                tenCotMoi = []
                for cot in khungDuLieu.columns:
                    if cot in demCot:
                        demCot[cot] += 1
                        tenCotMoi.append(f"{cot}_{demCot[cot]}")
                    else:
                        demCot[cot] = 1
                        tenCotMoi.append(cot)
                khungDuLieu.columns = tenCotMoi
                
                khungDuLieu = khungDuLieu[khungDuLieu['Player'].notna() & (khungDuLieu['Player'] != 'Player') & (khungDuLieu['Player'] != '')].copy()
                
                trungLap = khungDuLieu[khungDuLieu['Player'].duplicated(keep=False)][['Player', 'Squad', 'Pos']].drop_duplicates().values.tolist()
                if trungLap:
                    print(f"Cầu thủ trùng lặp trong {maBang}: {khungDuLieu[khungDuLieu['Player'].duplicated()]['Player'].tolist()}")
                    print(f"Chi tiết trùng lặp: {trungLap}")
                    khungDuLieu = khungDuLieu.groupby('Player').first().reset_index()
                
                print(f"Các cột trong bảng {maBang}: {khungDuLieu.columns.tolist()}")
                return khungDuLieu
            else:
                print(f"Không tìm thấy dữ liệu trong bảng tại {url}")
                return None
        
        except TimeoutException:
            print(f"Hết thời gian chờ bảng tại {url} (lần thử {lanThu + 1}/{soLanThuLai})")
        except NoSuchElementException:
            print(f"Bảng với ID {maBang} không tìm thấy tại {url} (lần thử {lanThu + 1}/{soLanThuLai})")
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu từ {url}: {e} (lần thử {lanThu + 1}/{soLanThuLai})")
        finally:
            trinhDieuKhien.quit()
        time.sleep(5)
    return None

def thucHienPhanCumVaPCA(khungDuLieu):
    """Thực hiện phân cụm K-means, xác định số cụm tối ưu, áp dụng PCA, và vẽ biểu đồ 2D."""
    print("Thực hiện phân cụm K-means và trực quan hóa PCA...")
    
    cotThuMon = ['GA90', 'Save%', 'CS%', 'PK Save%']
    
    # Xác định các cột số
    danhSachCotSo = []
    for cot in khungDuLieu.columns:
        if cot in ['Player', 'Team', 'Position', 'Nation', 'First_Name']:
            continue
        khungDuLieu[cot] = pd.to_numeric(khungDuLieu[cot].replace('N/a', pd.NA), errors='coerce')
        if khungDuLieu[cot].dtype in ['int64', 'float64']:
            danhSachCotSo.append(cot)
    
    # Chuẩn bị dữ liệu cho phân cụm
    duLieuPhanCum = khungDuLieu[danhSachCotSo].copy()
    
    # Loại bỏ cột toàn là NaN
    cotBanDau = duLieuPhanCum.columns.tolist()
    duLieuPhanCum = duLieuPhanCum.dropna(axis=1, how='all')
    cotBiXoa = set(cotBanDau) - set(duLieuPhanCum.columns)
    if cotBiXoa:
        print(f"Các cột bị xóa do toàn NaN: {cotBiXoa}")
    
    # Kiểm tra xem còn cột số nào không
    if duLieuPhanCum.empty:
        raise ValueError("Không có cột số nào cho phân cụm sau khi xóa cột NaN. Kiểm tra quá trình lấy dữ liệu và 'results.csv'.")
    
    # Thay thế giá trị thiếu bằng trung bình cột
    duLieuPhanCum = duLieuPhanCum.fillna(duLieuPhanCum.mean())
    
    # Xác minh không còn NaN
    if duLieuPhanCum.isna().any().any():
        cotNaN = duLieuPhanCum.columns[duLieuPhanCum.isna().any()].tolist()
        raise ValueError(f"Dữ liệu vẫn chứa NaN sau khi thay thế trong các cột: {cotNaN}. Kiểm tra 'results.csv' để tìm vấn đề dữ liệu.")
    
    print(f"Các cột được sử dụng cho phân cụm: {duLieuPhanCum.columns.tolist()}")
    
    # Chuẩn hóa dữ liệu
    chuanHoa = StandardScaler()
    duLieuDaChuanHoa = chuanHoa.fit_transform(duLieuPhanCum)
    
    # Xác định số cụm tối ưu bằng Elbow Method và Silhouette Score
    tongKhoangCach = []
    diemSilhouette = []
    khoangK = range(2, 11)
    
    for k in khoangK:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(duLieuDaChuanHoa)
        tongKhoangCach.append(kmeans.inertia_)
        nhan = kmeans.labels_
        diemSilhouette.append(silhouette_score(duLieuDaChuanHoa, nhan))
    
    # Vẽ biểu đồ Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(khoangK, tongKhoangCach, 'bo-')
    plt.title('Phương pháp Elbow để chọn k tối ưu')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Tổng bình phương khoảng cách trong cụm (WCSS)')
    plt.savefig('elbow_plot.png')
    plt.close()
    
    # Vẽ biểu đồ Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(khoangK, diemSilhouette, 'bo-')
    plt.title('Điểm Silhouette so với số cụm')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Điểm Silhouette')
    plt.savefig('silhouette_plot.png')
    plt.close()
    
    # Chọn k tối ưu (dựa trên Elbow và Silhouette Score)
    kToiUu = khoangK[np.argmax(diemSilhouette)]
    print(f"Số cụm tối ưu được chọn: {kToiUu}")
    print("Lý do: Điểm Silhouette cao nhất, cho thấy các cụm được tách biệt tốt. Biểu đồ Elbow cũng được xem xét.")
    
    # Áp dụng K-means với k tối ưu
    kmeans = KMeans(n_clusters=kToiUu, random_state=42)
    nhanCum = kmeans.fit_predict(duLieuDaChuanHoa)
    
    # Thêm nhãn cụm vào DataFrame
    khungDuLieu['Nhom'] = nhanCum
    
    # Lưu kết quả phân cụm
    khungDuLieu[['Player', 'Team', 'Position', 'Nhom']].to_csv('clusters.csv', index=False)
    print("Kết quả phân cụm đã được lưu vào 'clusters.csv'")
    
    # Phân tích cụm
    print("\nPhân tích cụm:")
    for cum in range(kToiUu):
        khungCum = khungDuLieu[khungDuLieu['Nhom'] == cum]
        print(f"\nCụm {cum} (Số lượng: {len(khungCum)} cầu thủ):")
        print(f"Trung bình Bàn thắng: {khungCum['Goals'].mean():.2f}")
        print(f"Trung bình Kiến tạo: {khungCum['Assists'].mean():.2f}")
        print(f"Trung bình Tkl: {khungCum['Tkl'].mean():.2f}")
        print(f"Trung bình Recov: {khungCum['Recov'].mean():.2f}")
        print(f"Phân bố vị trí:\n{khungCum['Position'].value_counts()}")
        print(f"Một số cầu thủ mẫu:\n{khungCum[['Player', 'Team', 'Position']].head(3)}")
    
    # Áp dụng PCA để giảm xuống 2 chiều
    pca = PCA(n_components=2)
    duLieuPCA = pca.fit_transform(duLieuDaChuanHoa)
    tiLeBienDo = pca.explained_variance_ratio_.sum()
    print(f"\nPCA: Tỷ lệ phương sai giải thích (2 thành phần): {tiLeBienDo:.2f}")
    
    # Vẽ biểu đồ 2D
    plt.figure(figsize=(10, 8))
    diem = plt.scatter(duLieuPCA[:, 0], duLieuPCA[:, 1], c=nhanCum, cmap='viridis', alpha=0.6)
    plt.colorbar(diem, label='Cụm')
    plt.title('Phân cụm 2D PCA của cầu thủ Premier League')
    plt.xlabel('Thành phần PCA 1')
    plt.ylabel('Thành phần PCA 2')
    
    # Gắn nhãn một số cầu thủ đại diện
    for cum in range(kToiUu):
        diemCum = duLieuPCA[nhanCum == cum]
        khungCum = khungDuLieu[nhanCum == cum]
        # Chọn cầu thủ đại diện (ví dụ: người có số bàn thắng cao nhất trong cụm)
        if not khungCum.empty:
            cauThuDinh = khungCum.sort_values(by='Goals', ascending=False).iloc[0]
            chiSo = khungCum.index[khungCum['Player'] == cauThuDinh['Player']].tolist()[0]
            chiSo = khungDuLieu.index.get_loc(chiSo)
            plt.annotate(cauThuDinh['Player'], (duLieuPCA[chiSo, 0], duLieuPCA[chiSo, 1]), fontsize=8)
    
    plt.savefig('clusters_2d.png')
    plt.close()
    print("Biểu đồ phân cụm 2D PCA đã được lưu vào 'clusters_2d.png'")

def chinh():
    """Chạy chính chương trình."""
    print("Danh sách bảng trước khi kiểm tra:")
    for i, muc in enumerate(danhSachURL):
        print(f"Chỉ số {i}: {muc}")
    
    print("\nĐang kiểm tra danh sách bảng...")
    for i, muc in enumerate(danhSachURL):
        if not isinstance(muc, tuple) or len(muc) != 2:
            raise ValueError(f"Mục bảng không hợp lệ tại chỉ số {i}: {muc}. Đòi hỏi tuple (url, maBang).")
    print("Danh sách bảng đã được kiểm tra thành công.")

    urlChuan, maBangChuan = danhSachURL[0]
    khungDuLieuChinh = layDuLieuBang(urlChuan, maBangChuan)
    if khungDuLieuChinh is None:
        print("Lấy dữ liệu thống kê chuẩn thất bại. Thoát.")
        return

    if 'Player' not in khungDuLieuChinh.columns:
        print("Không tìm thấy cột 'Player' trong bảng thống kê chuẩn. Thoát.")
        return

    khungDuLieuChinh['Player'] = khungDuLieuChinh['Player'].apply(lamSachTenCauThu)
    khungDuLieuChinh['Min'] = pd.to_numeric(khungDuLieuChinh['Min'].str.replace(',', ''), errors='coerce').fillna(0)
    khungDuLieuChinh = khungDuLieuChinh[khungDuLieuChinh['Min'] > 90].copy()
    khungDuLieuChinh['TenDau'] = khungDuLieuChinh['Player'].apply(layTenDau)

    ketQuaKhung = pd.DataFrame({'Player': khungDuLieuChinh['Player'], 'TenDau': khungDuLieuChinh['TenDau']})

    cotChuaAp = []
    for tenHienThi, cotFbref in bangThongKeCanThiet.items():
        if cotFbref in khungDuLieuChinh.columns:
            ketQuaKhung[tenHienThi] = khungDuLieuChinh[cotFbref].fillna('N/a')
            print(f"Đã ánh xạ {tenHienThi} tới {cotFbref} từ thống kê chuẩn")
        else:
            ketQuaKhung[tenHienThi] = 'N/a'
            cotChuaAp.append((tenHienThi, cotFbref, 'stats_standard'))
            print(f"Cột {cotFbref} không tìm thấy trong thống kê chuẩn cho {tenHienThi}")

    for url, maBang in danhSachURL[1:]:
        khung = layDuLieuBang(url, maBang)
        if khung is None or 'Player' not in khung.columns:
            print(f"Bỏ qua {url} do thiếu bảng hoặc cột 'Player'")
            continue
        khung['Player'] = khung['Player'].apply(lamSachTenCauThu)
        for tenHienThi, cotFbref in bangThongKeCanThiet.items():
            if cotFbref in khung.columns:
                khungTam = khung[['Player', cotFbref]].drop_duplicates('Player')
                duLieuGop = ketQuaKhung[['Player']].merge(
                    khungTam,
                    on='Player',
                    how='left'
                )[cotFbref].fillna('N/a')
                if ketQuaKhung[tenHienThi].eq('N/a').all():
                    ketQuaKhung[tenHienThi] = duLieuGop
                    print(f"Đã ánh xạ {tenHienThi} tới {cotFbref} từ {maBang}")
                else:
                    print(f"Bỏ qua {tenHienThi} vì đã được ánh xạ")
            else:
                if ketQuaKhung[tenHienThi].eq('N/a').all():
                    cotChuaAp.append((tenHienThi, cotFbref, maBang))
                print(f"Cột {cotFbref} không tìm thấy trong {maBang} cho {tenHienThi}")

    if cotChuaAp:
        print("\nCác cột chưa được ánh xạ:")
        for tenHienThi, cotFbref, maBang in cotChuaAp:
            print(f"{tenHienThi} ({cotFbref}) không tìm thấy trong {maBang}")

    thucHienPhanCumVaPCA(ketQuaKhung)

if __name__ == "__main__":
    chinh()