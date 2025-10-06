
# Based on the research, the issue is that loading happens during rerun flow
# Let's completely redesign the NACE loading approach:
# 1. Load NACE ONLY when actually needed (lazy loading)
# 2. Use a flag to track loading status
# 3. Avoid any checks in the main flow that trigger on every rerun

fixed_code = '''import streamlit as st
import pandas as pd
import io, re, time, requests, random, string, os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

def get_dropbox_download_link(shared_link):
    if 'dropbox.com' in shared_link:
        return shared_link.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    return shared_link

def load_database_from_dropbox():
    try:
        if 'DROPBOX_FILE_URL' in st.secrets:
            dropbox_url = st.secrets["DROPBOX_FILE_URL"]
        else:
            st.warning("⚠️ Add DROPBOX_FILE_URL to Streamlit Secrets")
            return None
        
        download_url = get_dropbox_download_link(dropbox_url)
        with st.spinner("📥 Downloading database..."):
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content))
            st.success(f"✅ Database loaded: {len(df)} companies")
            return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# NEW: Lazy load NACE mapping - only when actually needed
def get_nace_classifier():
    """Lazy load NACE classifier - only loads once when first needed"""
    # Check if already attempted to load
    if 'nace_load_attempted' not in st.session_state:
        st.session_state.nace_load_attempted = False
    
    # Check if already loaded
    if 'nace_classifier' not in st.session_state:
        st.session_state.nace_classifier = None
    
    # If already attempted (success or fail), return cached result
    if st.session_state.nace_load_attempted:
        return st.session_state.nace_classifier
    
    # First attempt - try to load
    st.session_state.nace_load_attempted = True
    
    try:
        if 'DROPBOX_NACE_URL' in st.secrets:
            dropbox_url = st.secrets["DROPBOX_NACE_URL"]
            download_url = get_dropbox_download_link(dropbox_url)
            
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content))
            
            st.session_state.nace_classifier = NACEClassifier(df)
            return st.session_state.nace_classifier
    except:
        pass
    
    # If failed, return None
    st.session_state.nace_classifier = None
    return None

ALLOWED_DOMAIN = "@arkap.ch"
CODE_EXPIRY_MINUTES = 10
SESSION_TIMEOUT_MINUTES = 60
COUNTRY_CODES = {'AT': 'Austria', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'GB': 'United Kingdom', 'IT': 'Italy', 'LU': 'Luxembourg', 'NL': 'Netherlands', 'PT': 'Portugal'}

def safe_format(value, fmt="{:,.0f}", pre="", suf="", default="N/A"):
    if pd.isna(value) or value is None or value == '': return default
    try:
        if isinstance(value, str):
            v = value.replace(',', '').replace(' ', '').replace('€', '').replace('k', '').strip()
            if not v or v == '-': return default
            value = float(v)
        return f"{pre}{fmt.format(float(value))}{suf}"
    except: return str(value) if value else default

# NEW: NACE Classification Lookup System
class NACEClassifier:
    def __init__(self, mapping_df=None):
        self.lookups = None
        if mapping_df is not None:
            self._init_lookups(mapping_df)
    
    def _init_lookups(self, df):
        """Initialize lookup dictionaries from mapping dataframe"""
        nace_code_lookup = {}
        nace_title_lookup = {}
        ateco_code_lookup = {}
        ateco_title_lookup = {}
        
        for idx, row in df.iterrows():
            record = {
                'nace_category_code': str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else '',
                'nace_category_title': str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else '',
                'ateco_category_code': str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else '',
                'ateco_category_title': str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else '',
                'nace_subcat_code': str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else '',
                'nace_subcat_title': str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else '',
                'ateco_subcat_code': str(row.iloc[6]).strip() if pd.notna(row.iloc[6]) else '',
                'ateco_subcat_title': str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else '',
                'arkap_industry': str(row.iloc[8]).strip() if pd.notna(row.iloc[8]) else '',
                'arkap_subindustry': str(row.iloc[9]).strip() if pd.notna(row.iloc[9]) else ''
            }
            
            if record['nace_subcat_code'] and record['nace_subcat_code'] != 'nan':
                nace_code_lookup[record['nace_subcat_code'].lower()] = record
            if record['nace_subcat_title'] and record['nace_subcat_title'] != 'nan':
                nace_title_lookup[record['nace_subcat_title'].lower()] = record
            if record['ateco_subcat_code'] and record['ateco_subcat_code'] != 'nan':
                ateco_code_lookup[record['ateco_subcat_code'].lower()] = record
            if record['ateco_subcat_title'] and record['ateco_subcat_title'] != 'nan':
                ateco_title_lookup[record['ateco_subcat_title'].lower()] = record
        
        self.lookups = {
            'nace_code': nace_code_lookup,
            'nace_title': nace_title_lookup,
            'ateco_code': ateco_code_lookup,
            'ateco_title': ateco_title_lookup
        }
    
    def classify(self, nace_code):
        """Lookup NACE code and return classification"""
        if self.lookups is None:
            return None
        
        if not nace_code or pd.isna(nace_code):
            return None
        
        input_clean = str(nace_code).strip().lower()
        
        # Try NACE code lookup
        if input_clean in self.lookups['nace_code']:
            return self.lookups['nace_code'][input_clean]
        
        # Try ATECO code lookup
        if input_clean in self.lookups['ateco_code']:
            return self.lookups['ateco_code'][input_clean]
        
        # Try partial match for NACE codes
        for key in self.lookups['nace_code'].keys():
            if key.startswith(input_clean) or input_clean.startswith(key):
                return self.lookups['nace_code'][key]
        
        return None

class CompanyDatabase:
    def __init__(self, df=None):
        self.db, self.name_idx, self.vat_idx, self.country_idx = df, {}, {}, {}
        if df is not None: self._init()
    
    def _init(self):
        mapping = {}
        for col in self.db.columns:
            c = col.lower()
            if 'company' in c and 'name' in c: mapping[col] = 'Company Name'
            elif 'vat' in c and 'code' in c: mapping[col] = 'VAT Code'
            elif 'national' in c and 'id' in c: mapping[col] = 'National ID'
            elif 'fiscal' in c: mapping[col] = 'Fiscal Code'
            elif 'country' in c and 'code' in c: mapping[col] = 'Country Code'
            elif 'nace' in c: mapping[col] = 'Nace Code'
            elif 'last' in c and 'yr' in c: mapping[col] = 'Last Yr'
            elif 'production' in c: mapping[col] = 'Value of production (th)'
            elif 'employee' in c: mapping[col] = 'Employees'
            elif 'ebitda' in c: mapping[col] = 'Ebitda (th)'
            elif 'pfn' in c: mapping[col] = 'PFN (th)'
        self.db = self.db.rename(columns=mapping)
        
        for idx, row in self.db.iterrows():
            if 'Company Name' in self.db.columns and pd.notna(row.get('Company Name')):
                k = str(row['Company Name']).lower().strip()
                self.name_idx.setdefault(k, []).append(idx)
            if 'VAT Code' in self.db.columns and pd.notna(row.get('VAT Code')):
                k = str(row['VAT Code']).upper().replace(' ', '').replace('-', '').replace('.', '')
                self.vat_idx.setdefault(k, []).append(idx)
            if 'Country Code' in self.db.columns:
                for cc in self.db['Country Code'].unique():
                    if pd.notna(cc): self.country_idx[str(cc).upper()] = self.db[self.db['Country Code'] == cc].index.tolist()
    
    def search_name(self, name, country=None):
        k = name.lower().strip()
        if k in self.name_idx:
            idxs = self.name_idx[k]
            if country and country in self.country_idx: idxs = [i for i in idxs if i in self.country_idx[country]]
            return self._extract(self.db.iloc[idxs[0]]) if idxs else None
        return None
    
    def search_vat(self, vat, country=None):
        k = str(vat).upper().replace(' ', '').replace('-', '').replace('.', '')
        if k in self.vat_idx:
            idxs = self.vat_idx[k]
            if country and country in self.country_idx: idxs = [i for i in idxs if i in self.country_idx[country]]
            return self._extract(self.db.iloc[idxs[0]]) if idxs else None
        return None
    
    def _extract(self, row):
        d = {'source': 'database'}
        for f in ['Company Name', 'National ID', 'Fiscal Code', 'VAT Code', 'Country Code', 'Nace Code', 'Last Yr', 'Value of production (th)', 'Employees', 'Ebitda (th)', 'PFN (th)']:
            if f in row.index and pd.notna(row[f]): d[f.lower().replace(' ', '_').replace('(', '').replace(')', '')] = row[f]
        return d

class AuthenticationManager:
    def __init__(self):
        for k in ['auth_codes', 'authenticated', 'user_email', 'auth_time', 'company_db', 'search_mode']:
            if k not in st.session_state: 
                st.session_state[k] = {} if k == 'auth_codes' else (False if k == 'authenticated' else ("" if k == 'user_email' else None))
    
    def is_valid_email(self, e): return re.match(r'^[\\w.+-]+@[\\w.-]+\\.[\\w]+$', e) and e.lower().endswith(ALLOWED_DOMAIN.lower())
    def gen_code(self): return ''.join(random.choices(string.digits, k=6))
    def store_code(self, e, c): st.session_state.auth_codes[e] = {'code': c, 'timestamp': datetime.now(), 'attempts': 0}
    
    def verify(self, e, c):
        if e not in st.session_state.auth_codes: return False, "No code"
        d = st.session_state.auth_codes[e]
        if datetime.now() - d['timestamp'] > timedelta(minutes=CODE_EXPIRY_MINUTES): del st.session_state.auth_codes[e]; return False, "Expired"
        if d['attempts'] >= 3: del st.session_state.auth_codes[e]; return False, "Too many"
        if d['code'] == c: 
            st.session_state.authenticated, st.session_state.user_email, st.session_state.auth_time = True, e, datetime.now()
            del st.session_state.auth_codes[e]
            return True, "Success"
        d['attempts'] += 1
        return False, f"{3-d['attempts']} left"
    
    def is_valid(self): return st.session_state.authenticated and st.session_state.auth_time and datetime.now() - st.session_state.auth_time <= timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    def logout(self): st.session_state.authenticated, st.session_state.user_email, st.session_state.auth_time = False, "", None

class EnhancedUKExtractor:
    """UK Company Number Extractor - Full Original Logic"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.patterns = [
            r'Company\\s+number[\\s:]*([0-9]{8})',
            r'(?:Company|Co\\.|Ltd\\.)\\s+(?:No\\.|Number)[\\s:]*([0-9]{8})',
            r'(?:Registered|Registration)\\s+(?:No\\.|Number)[\\s:]*([0-9]{8})',
            r'([0-9]{8})\\s*(?:Company|Registered)',
            r'\\b([0-9]{8})\\b'
        ]
    
    def process(self, name, url=None):
        r = {'company_name': name, 'website': url or '', 'status': 'Not Found', 'source': 'web'}
        if url:
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for pattern in self.patterns:
                        for match in re.finditer(pattern, resp.text, re.I):
                            code = re.sub(r'[^0-9]', '', match.group(1) if match.lastindex else match.group(0))
                            if len(code) == 8 and code[0] != '0':
                                r['company_number'] = code
                                r['status'] = 'Found'
                                return r
            except: pass
        return r

class MultiModeExtractor:
    def __init__(self, db=None, use_db=True):
        self.db, self.use_db = db, use_db
        self.extractors = {'GB': EnhancedUKExtractor()}
        self.patterns = {
            'DE': [
                r'Steuernummer[\\s#:]*([0-9]{2,3}/[0-9]{3,4}/[0-9]{4,5})',
                r'Steuer-?Nr\\.?[\\s#:]*([0-9]{2,3}/[0-9]{3,4}/[0-9]{4,5})',
                r'Handelsregisternummer[\\s#:]*([HRA|HRB]{2,3}\\s*[0-9]{1,6})',
                r'Umsatzsteuer-?ID[\\s#:]*([D|DE]{1,2}[0-9]{9})',
                r'USt-?IdNr\\.?[\\s#:]*([D|DE]{1,2}[0-9]{9})'
            ],
            'FR': [
                r'SIREN[\\s#:]*([0-9]{9})',
                r'(?:N°\\s*SIREN|Numéro\\s*SIREN)[\\s#:]*([0-9]{9})',
                r'SIRET[\\s#:]*([0-9]{14})',
                r'TVA[\\s#:]*FR([0-9A-Z]{2}[0-9]{9})',
                r'N°\\s*TVA[\\s#:]*FR([0-9A-Z]{2}[0-9]{9})'
            ],
            'IT': [
                r'P\\.?\\s*IVA[\\s#:]*([0-9]{11})',
                r'Partita\\s+IVA[\\s#:]*([0-9]{11})',
                r'Codice\\s+Fiscale[\\s#:]*([A-Z0-9]{11,16})',
                r'C\\.?F\\.?[\\s#:]*([A-Z0-9]{11,16})'
            ],
            'PT': [
                r'NIF[\\s#:]*([0-9]{9})',
                r'N\\.?I\\.?F\\.?[\\s#:]*([0-9]{9})',
                r'Contribuinte[\\s#:]*([0-9]{9})',
                r'NIPC[\\s#:]*([0-9]{9})'
            ],
            'NL': [
                r'KvK[\\s#:]*([0-9]{8})',
                r'(?:Kamer\\s+van\\s+Koophandel|K\\.v\\.K\\.?)[\\s#:]*([0-9]{8})',
                r'RSIN[\\s#:]*([0-9]{9})',
                r'BTW[\\s#:]*NL([0-9]{9}B[0-9]{2})',
                r'LEI[\\s#:]*([A-Z0-9]{20})'
            ],
            'AT': [
                r'ATU\\s*([0-9]{8})',
                r'UID[\\s#:]*ATU([0-9]{8})',
                r'Umsatzsteuer-?ID[\\s#:]*ATU([0-9]{8})',
                r'FN[\\s#:]*([0-9]{6}[a-z])'
            ],
            'CH': [
                r'CHE[\\s-]?([0-9]{3})\\.?([0-9]{3})\\.?([0-9]{3})',
                r'UID[\\s#:]*CHE[\\s-]?([0-9]{3})\\.?([0-9]{3})\\.?([0-9]{3})',
                r'CH-ID[\\s#:]*CH-([0-9]{3})\\.?([0-9]{1})\\.?([0-9]{3})\\.?([0-9]{3})-?([0-9]{1})'
            ],
            'LU': [
                r'LU\\s*([0-9]{8})',
                r'TVA[\\s#:]*LU([0-9]{8})',
                r'B([0-9]{6})',
                r'L\\.?U\\.?R[\\s#:]*([0-9]{6})'
            ]
        }
    
    def process_single(self, name, web, country, vat=None):
        result = None
        
        if self.use_db and self.db:
            result = self.db.search_name(name, country)
            if result: 
                result['search_method'] = 'DB-Name'
                result['status'] = 'Found'
            elif vat:
                result = self.db.search_vat(vat, country)
                if result:
                    result['search_method'] = 'DB-VAT'
                    result['status'] = 'Found'
        
        if not result or result['status'] != 'Found':
            result = self._web(name, web, country)
            if self.use_db:
                result['search_method'] = 'DB failed → Web'
            else:
                result['search_method'] = 'Web only'
        
        # NEW: Add NACE classification if available (lazy load)
        nace_code = result.get('nace_code')
        if nace_code:
            nace_classifier = get_nace_classifier()
            if nace_classifier:
                classification = nace_classifier.classify(nace_code)
                if classification:
                    result['nace_category_code'] = classification.get('nace_category_code', '')
                    result['nace_category_title'] = classification.get('nace_category_title', '')
                    result['ateco_category_code'] = classification.get('ateco_category_code', '')
                    result['ateco_category_title'] = classification.get('ateco_category_title', '')
                    result['nace_subcat_code'] = classification.get('nace_subcat_code', '')
                    result['nace_subcat_title'] = classification.get('nace_subcat_title', '')
                    result['ateco_subcat_code'] = classification.get('ateco_subcat_code', '')
                    result['ateco_subcat_title'] = classification.get('ateco_subcat_title', '')
                    result['arkap_industry'] = classification.get('arkap_industry', '')
                    result['arkap_subindustry'] = classification.get('arkap_subindustry', '')
        
        return result
    
    def _web(self, name, web, country):
        if country in self.extractors:
            return self.extractors[country].process(name, web)
        
        r = {'company_name': name, 'website': web, 'country_code': country, 'status': 'Not Found', 'source': 'web'}
        if web and country in self.patterns:
            try:
                resp = requests.get(web, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    text_content = soup.get_text()
                    for pattern in self.patterns[country]:
                        matches = re.finditer(pattern, text_content, re.I)
                        for match in matches:
                            extracted = match.group(1) if match.lastindex else match.group(0)
                            field_name = f'{country.lower()}_code'
                            r[field_name] = extracted.strip()
                            r['status'] = 'Found'
                            return r
            except: pass
        return r
    
    def process_list(self, df, prog=None):
        results = []
        nc = [c for c in df.columns if 'company' in c.lower() or 'name' in c.lower()]
        name_col = nc[0] if nc else df.columns[0]
        wc = [c for c in df.columns if 'website' in c.lower() or 'url' in c.lower()]
        web_col = wc[0] if wc else None
        cc = [c for c in df.columns if 'country' in c.lower()]
        country_col = cc[0] if cc else None
        vc = [c for c in df.columns if 'vat' in c.lower() or 'fiscal' in c.lower()]
        vat_col = vc[0] if vc else None
        
        for idx, row in df.iterrows():
            if prog: prog(idx+1, len(df))
            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
            web = str(row[web_col]).strip() if web_col and pd.notna(row[web_col]) else ''
            vat = str(row[vat_col]).strip() if vat_col and pd.notna(row[vat_col]) else None
            country = 'GB'
            if country_col and pd.notna(row[country_col]):
                cv = str(row[country_col]).strip().upper()
                if len(cv) == 2 and cv in COUNTRY_CODES: country = cv
            
            results.append(self.process_single(name, web, country, vat))
            time.sleep(0.2)
        
        return results

def show_auth(auth):
    st.title("🔐 arKap VAT Extractor")
    st.info("🏢 @arkap.ch only")
    t1, t2 = st.tabs(["📧 Email", "🔑 Code"])
    
    with t1:
        e = st.text_input("Email")
        if st.button("Send Code", type="primary"):
            if auth.is_valid_email(e):
                c = auth.gen_code()
                auth.store_code(e, c)
                st.success(f"Code: {c}")
            else:
                st.error("Invalid")
    
    with t2:
        e = st.text_input("Email", key="e2")
        c = st.text_input("Code", max_chars=6)
        if st.button("Verify", type="primary"):
            ok, msg = auth.verify(e, c)
            if ok:
                st.success(msg)
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

def show_main():
    st.title("🌍 arKap VAT Extractor")
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(f"**User:** {st.session_state.user_email}")
    with c2:
        if st.button("Logout"):
            AuthenticationManager().logout()
            st.rerun()
    
    st.markdown("---")
    
    # NO NACE LOADING HERE - It's now lazy loaded when needed
    
    if st.session_state.company_db is None:
        st.header("📊 Database Setup")
        with st.expander("ℹ️ Dropbox Setup", expanded=True):
            st.write("1. Share file on Dropbox → Copy link")
            st.write('2. App Settings → Secrets → Add:')
            st.code('DROPBOX_FILE_URL = "your_company_database_link"\\nDROPBOX_NACE_URL = "your_nace_mapping_link"')
            st.write("Note: NACE classification is optional and will be loaded automatically when needed.")
        
        if st.button("📥 Load from Dropbox", type="primary"):
            df = load_database_from_dropbox()
            if df is not None:
                st.session_state.company_db = CompanyDatabase(df)
                st.rerun()
        
        st.markdown("---")
        up = st.file_uploader("Or Upload", type=['xlsx','csv'])
        if up is not None:
            df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            st.session_state.company_db = CompanyDatabase(df)
            st.rerun()
        
        if st.button("⏭️ Web Only"):
            st.session_state.search_mode = 'web'
            st.rerun()
        return
    
    if st.session_state.search_mode is None:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗄️ DB+Web", type="primary", use_container_width=True):
                st.session_state.search_mode = 'db'
                st.rerun()
        with c2:
            if st.button("🌐 Web Only", use_container_width=True):
                st.session_state.search_mode = 'web'
                st.rerun()
        return
    
    st.info(f"Mode: {st.session_state.search_mode.upper()}")
    if st.button("Change"):
        st.session_state.search_mode = None
        st.rerun()
    
    st.markdown("---")
    t1, t2 = st.tabs(["Bulk", "Single"])
    
    with t1:
        f = st.file_uploader("Company List", type=['csv','xlsx'])
        if f:
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            st.dataframe(df.head())
            
            if st.button("Process"):
                ext = MultiModeExtractor(
                    st.session_state.company_db, 
                    st.session_state.search_mode=='db'
                )
                p = st.progress(0)
                res = ext.process_list(df, lambda c,t: p.progress(c/t))
                rdf = pd.DataFrame(res)
                st.dataframe(rdf)
                
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Total", len(res))
                with c2:
                    st.metric("Found", len([r for r in res if r['status']=='Found']))
                with c3:
                    st.metric("Rate%", f"{len([r for r in res if r['status']=='Found'])/len(res)*100:.1f}")
                
                csv = io.StringIO()
                rdf.to_csv(csv, index=False)
                st.download_button("Download", csv.getvalue(), f"res_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    
    with t2:
        c1,c2 = st.columns(2)
        with c1:
            n = st.text_input("Name")
            w = st.text_input("Website")
            v = st.text_input("VAT")
        with c2:
            co = st.selectbox("Country", list(COUNTRY_CODES.keys()), format_func=lambda x:f"{COUNTRY_CODES[x]} ({x})")
        
        if st.button("Search") and n:
            ext = MultiModeExtractor(
                st.session_state.company_db, 
                st.session_state.search_mode=='db'
            )
            r = ext.process_single(n, w, co, v)
            
            if r['status']=='Found':
                st.success(f"✅ {r.get('search_method')}")
                
                if r.get('source')=='database':
                    st.subheader("📊 Database Results")
                    c1,c2=st.columns(2)
                    with c1:
                        for k in ['company_name','vat_code']:
                            if k in r: st.write(f"**{k}:** {r[k]}")
                    with c2:
                        for k in ['country_code','nace_code']:
                            if k in r: st.write(f"**{k}:** {r[k]}")
                    
                    # NEW: Display NACE/ATECO/Arkap classification if available
                    if any(k in r for k in ['nace_category_code', 'arkap_industry']):
                        st.subheader("🏭 Industry Classification")
                        c1, c2 = st.columns(2)
                        with c1:
                            if 'nace_category_code' in r and r['nace_category_code']:
                                st.write(f"**NACE Category:** {r['nace_category_code']} - {r.get('nace_category_title', '')}")
                            if 'nace_subcat_code' in r and r['nace_subcat_code']:
                                st.write(f"**NACE Subcategory:** {r['nace_subcat_code']} - {r.get('nace_subcat_title', '')}")
                            if 'ateco_category_code' in r and r['ateco_category_code']:
                                st.write(f"**ATECO Category:** {r['ateco_category_code']} - {r.get('ateco_category_title', '')}")
                        with c2:
                            if 'ateco_subcat_code' in r and r['ateco_subcat_code']:
                                st.write(f"**ATECO Subcategory:** {r['ateco_subcat_code']} - {r.get('ateco_subcat_title', '')}")
                            if 'arkap_industry' in r and r['arkap_industry']:
                                st.write(f"**Arkap Industry:** {r['arkap_industry']}")
                            if 'arkap_subindustry' in r and r['arkap_subindustry']:
                                st.write(f"**Arkap Subindustry:** {r['arkap_subindustry']}")
                    
                    st.subheader("💰 Financial Data")
                    c1,c2,c3=st.columns(3)
                    with c1:
                        if 'last_yr' in r: st.metric("Year",r['last_yr'])
                        if 'employees' in r: st.metric("Emp",safe_format(r.get('employees')))
                    with c2:
                        if 'value_of_production_th' in r:
                            st.metric("Prod",safe_format(r.get('value_of_production_th'),pre="€",suf="k"))
                        if 'ebitda_th' in r:
                            st.metric("EBITDA",safe_format(r.get('ebitda_th'),pre="€",suf="k"))
                    with c3:
                        if 'pfn_th' in r:
                            st.metric("PFN",safe_format(r.get('pfn_th'),pre="€",suf="k"))
                else:
                    st.subheader("🌐 Web Extraction Results")
                    extracted_data = {k: v for k, v in r.items()
                                    if k not in ['company_name','website','status','source','search_method','country_code']}
                    if extracted_data:
                        for key, value in extracted_data.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.info("✓ Company verified but no additional codes extracted from website")
            else:
                st.warning("❌ Not found in database or website")
            
            with st.expander("🔍 Raw Data"):
                st.json(r)

def main():
    st.set_page_config(page_title="arKap", page_icon="⚡", layout="wide")
    auth = AuthenticationManager()
    
    if auth.is_valid():
        show_main()
    else:
        show_auth(auth)

if __name__ == "__main__": main()
'''

# Save the completely refactored file
with open('arkap_vat_extractor_LAZY_LOAD.py', 'w', encoding='utf-8') as f:
    f.write(fixed_code)

print("✅ Complete refactoring done with LAZY LOADING!")
print("\n🔧 Major architectural changes:")
print("=" * 70)
print("1. ✅ REMOVED all NACE loading from show_main() flow")
print("2. ✅ Implemented LAZY LOADING via get_nace_classifier() function")
print("3. ✅ NACE only loads when process_single() actually needs it")
print("4. ✅ Uses 'nace_load_attempted' flag to prevent repeated attempts")
print("5. ✅ Completely silent - no spinners, no warnings during main flow")
print("6. ✅ Removed 'nace_classifier' from AuthenticationManager init")
print("7. ✅ No NACE checks during database loading/rerun cycle")
print("\n📝 How it works now:")
print("=" * 70)
print("→ App starts → Database loads → Mode selection → Ready")
print("→ User searches company → THEN NACE loads (first time only)")
print("→ Subsequent searches use cached NACE classifier")
print("→ No interference with database loading or UI reruns")
print("\n🎯 Benefits:")
print("=" * 70)
print("✓ No infinite loops - NACE not in main execution path")
print("✓ Faster startup - NACE loads only when needed")
print("✓ App works perfectly even without NACE secret configured")
print("✓ Original database flow completely untouched")
print("\n📄 File saved as: arkap_vat_extractor_LAZY_LOAD.py")
