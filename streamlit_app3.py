# son guncel calısan github
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import math
import sqlite3
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import base64
import requests
import xml.etree.ElementTree as ET

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the original agent code components
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    RFE,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import get_scorer

# ---- Enhanced PubMed Integration with Article Details ----
class SimplePubMedSearcher:
    """Enhanced PubMed searcher with rate limiting and article fetching"""
    
    def __init__(self, email: str, api_key: Optional[str] = None, delay: float = 0.34):
        self.email = email
        self.api_key = api_key
        self.delay = delay if not api_key else 0.1  # Faster with API key
        self.last_request = 0
        self.cache = {}
        self.article_cache = {}  # Cache for article details
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        time_since_last = time.time() - self.last_request
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request = time.time()
    
    def fetch_article_details(self, pmids: List[str]) -> List[dict]:
        """Fetch article details from PubMed IDs"""
        if not pmids:
            return []
        
        # Check cache first
        articles = []
        pmids_to_fetch = []
        for pmid in pmids:
            if pmid in self.article_cache:
                articles.append(self.article_cache[pmid])
            else:
                pmids_to_fetch.append(pmid)
        
        if not pmids_to_fetch:
            return articles
        
        self._rate_limit()
        
        try:
            # Fetch article details using efetch
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids_to_fetch),
                'retmode': 'xml',
                'email': self.email,
                'tool': 'FeatureSelectionAgent'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = requests.get(fetch_url, params=params, timeout=15)
            root = ET.fromstring(response.content)
            
            # Parse each article
            for article_elem in root.findall('.//PubmedArticle'):
                article_info = self._parse_article(article_elem)
                if article_info:
                    articles.append(article_info)
                    self.article_cache[article_info['pmid']] = article_info
            
        except Exception as e:
            st.warning(f"Error fetching article details: {str(e)}")
        
        return articles
    
    def _parse_article(self, article_elem) -> dict:
        """Parse article XML element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if fore_name is not None:
                        author_name = f"{author_name} {fore_name.text[0]}"
                    authors.append(author_name)
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            # Extract year
            year_elem = article_elem.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else "Unknown"
            
            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
        except Exception:
            return None
    
    def search_simple(self, feature_name: str, disease_context: str = None, max_results: int = 5) -> dict:
        """Enhanced PubMed search with article details"""
        cache_key = f"{feature_name}_{disease_context}_{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Build search query
            query_parts = [f'"{feature_name}"']
            if disease_context:
                query_parts.append(f'"{disease_context}"')
            query_parts.extend(["biomarker", "expression", "association"])
            search_term = " AND ".join(query_parts)
            
            # PubMed E-utilities search
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': search_term,
                'retmax': max_results,
                'email': self.email,
                'tool': 'FeatureSelectionAgent',
                'sort': 'relevance'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = requests.get(search_url, params=params, timeout=10)
            root = ET.fromstring(response.content)
            
            # Extract IDs and count
            ids = [id_elem.text for id_elem in root.findall('.//Id')]
            count_elem = root.find('.//Count')
            count = int(count_elem.text) if count_elem is not None else 0
            
            # Calculate evidence score (0-5 scale)
            evidence_score = min(count / 20.0, 5.0)
            if disease_context and count > 0:
                evidence_score *= 1.2  # Bonus for disease context
            
            # Fetch article details for top results
            articles = self.fetch_article_details(ids[:5]) if ids else []
            
            result = {
                'feature_name': feature_name,
                'paper_count': count,
                'sample_ids': ids[:5],
                'articles': articles,  # Now includes full article details
                'evidence_score': round(evidence_score, 1),
                'search_query': search_term,
                'disease_context': disease_context
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            st.warning(f"PubMed search error for {feature_name}: {str(e)}")
            return {
                'feature_name': feature_name,
                'paper_count': 0,
                'sample_ids': [],
                'articles': [],
                'evidence_score': 0.0,
                'search_query': '',
                'disease_context': disease_context
            }
    
    def batch_search(self, features: List[str], disease_context: str = None, 
                    progress_callback=None) -> List[dict]:
        """Batch search for multiple features"""
        results = []
        total = len(features)
        
        for i, feature in enumerate(features):
            if progress_callback:
                progress_callback(i + 1, total, feature)
            
            result = self.search_simple(feature, disease_context)
            results.append(result)
    """Enhanced PubMed searcher with rate limiting and article fetching"""
    
    def __init__(self, email: str, api_key: Optional[str] = None, delay: float = 0.34):
        self.email = email
        self.api_key = api_key
        self.delay = delay if not api_key else 0.1  # Faster with API key
        self.last_request = 0
        self.cache = {}
        self.article_cache = {}  # Cache for article details
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        time_since_last = time.time() - self.last_request
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request = time.time()
    
def generate_heart_df(n_samples=1000, seed=42, add_missing=True):
    rng = np.random.default_rng(seed)

    # --- Age & Sex ---
    age = rng.integers(25, 81, size=n_samples)
    # 0=female, 1=male (birçok klinik veri setinde erkek oranı daha yüksek görülür)
    sex = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])

    # --- Blood pressure (mmHg) ---
    # Temel: yaş + cinsiyet etkisi + gürültü
    sbp_mu = 110 + 0.6*age + 5*sex
    resting_blood_pressure = np.clip(np.round(rng.normal(sbp_mu, 12)), 85, 220).astype(int)

    # --- Serum cholesterol (mg/dL) ---
    chol_mu = 170 + 0.9*age + 10*sex
    serum_cholesterol = np.clip(np.round(rng.normal(chol_mu, 35)), 100, 400).astype(int)

    # --- Fasting blood sugar (>120mg/dl: 1) ---
    # Yaş ve kolesterolle ılımlı ilişki
    fbs_p = 1/(1 + np.exp(-(-5 + 0.03*age + 0.005*(serum_cholesterol-200))))
    fasting_blood_sugar = rng.binomial(1, fbs_p)

    # --- Resting ECG (0: normal, 1: ST-T anorm., 2: sol VH) ---
    # Hipertansiyon ve yaş ECG anormallik olasılığını artırır
    ecg_p1 = np.clip(0.15 + 0.003*(resting_blood_pressure-120) + 0.002*(age-50), 0, 0.6)
    ecg_p2 = np.clip(0.03 + 0.002*(resting_blood_pressure-130), 0, 0.2)
    ecg_p0 = np.clip(1 - ecg_p1 - ecg_p2, 0.05, 1)
    probs = np.vstack([ecg_p0, ecg_p1, ecg_p2]).T
    probs = probs / probs.sum(1, keepdims=True)
    resting_ecg_results = np.array([rng.choice([0,1,2], p=p) for p in probs])

    # --- Chest pain type (0:typical angina,1:atypical,2:non-anginal,3:asymptomatic) ---
    # Hastalık riski arttıkça 3 (asemptomatik) ve 0 (tipik) biraz daha olası
    base_cpt = np.array([0.30, 0.20, 0.35, 0.15])
    chest_pain_type = rng.choice([0,1,2,3], size=n_samples, p=base_cpt)

    # --- Max heart rate achieved ---
    hr_pred = 208 - 0.7*age + rng.normal(0, 10, n_samples)
    max_heart_rate_achieved = np.clip(np.round(hr_pred), 90, 210).astype(int)

    # --- Exercise-induced angina ---
    # Daha yüksek SBP/kolesterol, daha düşük HR kapasitesi => daha yüksek olasılık
    angina_logit = -2.0 + 0.01*(resting_blood_pressure-120) + 0.006*(serum_cholesterol-200) - 0.015*(max_heart_rate_achieved-150) + 0.4*(chest_pain_type==0) + 0.3*(chest_pain_type==3)
    angina_p = 1/(1+np.exp(-angina_logit))
    exercise_induced_angina = rng.binomial(1, np.clip(angina_p, 0.02, 0.8))

    # --- ST depression (oldpeak) ---
    st_base = np.maximum(0, rng.normal(0.6 + 0.01*(age-50) + 0.008*(resting_blood_pressure-120), 0.6, n_samples))
    st_depression = np.round(st_base + 0.8*exercise_induced_angina + 0.3*(resting_ecg_results==1), 1)

    # --- ST slope (0:down,1:flat,2:up) ---
    # Daha yüksek ST depresyonu -> down/flat olasılığı artar
    slope_scores = np.clip(st_depression, 0, None)
    p_down = np.clip(0.05 + 0.10*slope_scores, 0.05, 0.60)
    p_flat = np.clip(0.30 + 0.10*slope_scores, 0.20, 0.70)
    p_up = np.clip(1 - p_down - p_flat, 0.05, 0.75)
    # normalize
    norm = (p_down + p_flat + p_up)
    p_down, p_flat, p_up = p_down/norm, p_flat/norm, p_up/norm
    st_slope = np.array([rng.choice([0,1,2], p=[pdn, pft, pup]) for pdn, pft, pup in zip(p_down, p_flat, p_up)])

    # --- Number of major vessels (0-3) ---
    # Hastalığı olanlarda 1-3 daha olası
    vessel_logits = -0.2 + 0.004*(age-50) + 0.003*(serum_cholesterol-200) + 0.25*exercise_induced_angina + 0.15*(st_depression>1.0)
    vessel_p = 1/(1+np.exp(-vessel_logits))
    number_of_major_vessels = rng.choice([0,1,2,3], size=n_samples, p=[np.mean(1-vessel_p), np.mean(0.5*vessel_p), np.mean(0.3*vessel_p), np.mean(0.2*vessel_p)])

    # --- Thalassemia type (1=fixed defect, 2=normal, 3=reversible defect) ---
    # Hastalıkla ilişki: 1 ve 3 bir miktar artar
    thal = []
    for i in range(n_samples):
        if st_depression[i] > 1.0 or exercise_induced_angina[i]==1:
            thal.append(rng.choice([1,2,3], p=[0.12, 0.25, 0.63]))
        else:
            thal.append(rng.choice([1,2,3], p=[0.06, 0.42, 0.52]))
    thalassemia_type = np.array(thal)

    # --- Disease probability (logistic model) ---
    logit = (
        -8.0
        + 0.045*(age-50)
        + 0.018*(resting_blood_pressure-120)
        + 0.012*(serum_cholesterol-200)
        + 0.6*fasting_blood_sugar
        + 0.5*exercise_induced_angina
        + 0.35*(resting_ecg_results==1)
        + 0.25*(resting_ecg_results==2)
        + 0.40*(st_depression)          # her 1 birim ↑ risk ↑
        + 0.45*(st_slope==0)            # downsloping
        + 0.25*(st_slope==1)            # flat
        + 0.35*(chest_pain_type==0)     # tipik anjina
        + 0.20*(chest_pain_type==3)     # asemptomatik
        + 0.20*sex
        + 0.30*(number_of_major_vessels>=1)
        + 0.15*(thalassemia_type==1)
        + 0.25*(thalassemia_type==3)
    )
    p_disease = 1/(1+np.exp(-logit))
    has_disease = rng.binomial(1, np.clip(p_disease, 0.02, 0.95))

    # Göğüs ağrısı tipinin dağılımını hastalıkla biraz uyumlu hale getir (opsiyonel)
    # Hastalığı olmayanlarda non-anginal (2) biraz daha sık, hastalığı olanlarda 0 ve 3 biraz artar
    adjust_mask = rng.random(n_samples) < 0.15
    chest_pain_type = chest_pain_type.copy()
    chest_pain_type[(has_disease==0) & adjust_mask] = rng.choice([1,2], size=((has_disease==0) & adjust_mask).sum(), p=[0.35, 0.65])
    chest_pain_type[(has_disease==1) & adjust_mask] = rng.choice([0,3], size=((has_disease==1) & adjust_mask).sum(), p=[0.55, 0.45])

    # --- küçük oranlarda aykırı değer ve eksik değer ekle (opsiyonel) ---
    if add_missing:
        miss_mask = rng.random(n_samples) < 0.02
        serum_cholesterol = serum_cholesterol.astype('float')
        bp_out_mask = rng.random(n_samples) < 0.01
        resting_blood_pressure[bp_out_mask] = np.clip(resting_blood_pressure[bp_out_mask] + rng.integers(30, 60, bp_out_mask.sum()), 120, 260)

    heart_df = pd.DataFrame({
        'age': age,
        'sex': sex,  # 0=female, 1=male
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,  # >120mg/dl
        'resting_ecg_results': resting_ecg_results,
        'max_heart_rate_achieved': max_heart_rate_achieved,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'st_slope': st_slope,
        'number_of_major_vessels': number_of_major_vessels,
        'thalassemia_type': thalassemia_type,
        'has_disease': has_disease,
        'disease_prob': p_disease.round(3)
    })

    # Türleri düzelt
    int_cols = ['age','sex','chest_pain_type','resting_blood_pressure','fasting_blood_sugar',
                'resting_ecg_results','max_heart_rate_achieved','exercise_induced_angina',
                'st_slope','number_of_major_vessels','thalassemia_type','has_disease']
    heart_df[int_cols] = heart_df[int_cols].astype(int)

    return heart_df
    def fetch_article_details(self, pmids: List[str]) -> List[dict]:
        """Fetch article details from PubMed IDs"""
        if not pmids:
            return []
        
        # Check cache first
        articles = []
        pmids_to_fetch = []
        for pmid in pmids:
            if pmid in self.article_cache:
                articles.append(self.article_cache[pmid])
            else:
                pmids_to_fetch.append(pmid)
        
        if not pmids_to_fetch:
            return articles
        
        self._rate_limit()
        
        try:
            # Fetch article details using efetch
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids_to_fetch),
                'retmode': 'xml',
                'email': self.email,
                'tool': 'FeatureSelectionAgent'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = requests.get(fetch_url, params=params, timeout=15)
            root = ET.fromstring(response.content)
            
            # Parse each article
            for article_elem in root.findall('.//PubmedArticle'):
                article_info = self._parse_article(article_elem)
                if article_info:
                    articles.append(article_info)
                    self.article_cache[article_info['pmid']] = article_info
            
        except Exception as e:
            st.warning(f"Error fetching article details: {str(e)}")
        
        return articles
    
    def _parse_article(self, article_elem) -> dict:
        """Parse article XML element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if fore_name is not None:
                        author_name = f"{author_name} {fore_name.text[0]}"
                    authors.append(author_name)
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            # Extract year
            year_elem = article_elem.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else "Unknown"
            
            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            
            return {
                'pmid': pmid,
                'title': title,
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
        except Exception:
            return None
    
    def search_simple(self, feature_name: str, disease_context: str = None, max_results: int = 5) -> dict:
        """Enhanced PubMed search with article details"""
        cache_key = f"{feature_name}_{disease_context}_{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Build search query
            query_parts = [f'"{feature_name}"']
            if disease_context:
                query_parts.append(f'"{disease_context}"')
            query_parts.extend(["biomarker", "expression", "association"])
            search_term = " AND ".join(query_parts)
            
            # PubMed E-utilities search
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': search_term,
                'retmax': max_results,
                'email': self.email,
                'tool': 'FeatureSelectionAgent',
                'sort': 'relevance'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            response = requests.get(search_url, params=params, timeout=10)
            root = ET.fromstring(response.content)
            
            # Extract IDs and count
            ids = [id_elem.text for id_elem in root.findall('.//Id')]
            count_elem = root.find('.//Count')
            count = int(count_elem.text) if count_elem is not None else 0
            
            # Calculate evidence score (0-5 scale)
            evidence_score = min(count / 20.0, 5.0)
            if disease_context and count > 0:
                evidence_score *= 1.2  # Bonus for disease context
            
            # Fetch article details for top results
            articles = self.fetch_article_details(ids[:5]) if ids else []
            
            result = {
                'feature_name': feature_name,
                'paper_count': count,
                'sample_ids': ids[:5],
                'articles': articles,  # Now includes full article details
                'evidence_score': round(evidence_score, 1),
                'search_query': search_term,
                'disease_context': disease_context
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            st.warning(f"PubMed search error for {feature_name}: {str(e)}")
            return {
                'feature_name': feature_name,
                'paper_count': 0,
                'sample_ids': [],
                'articles': [],
                'evidence_score': 0.0,
                'search_query': '',
                'disease_context': disease_context
            }
    
    def batch_search(self, features: List[str], disease_context: str = None, 
                    progress_callback=None) -> List[dict]:
        """Batch search for multiple features"""
        results = []
        total = len(features)
        
        for i, feature in enumerate(features):
            if progress_callback:
                progress_callback(i + 1, total, feature)
            
            result = self.search_simple(feature, disease_context)
            results.append(result)
            
        return results

# ---- Agent Code (embedded for Streamlit) ----
@dataclass
class TrialPlan:
    strategy: str
    params: Dict[str, Any]
    comment: str = ""

@dataclass
class TrialResult:
    metric_name: str
    metric_value: float
    metric_std: float
    n_features: int
    selected_features: List[str]
    pipeline_repr: str
    duration_sec: float
    reflection: str = ""

@dataclass
class AgentConfig:
    target_metric: str = "roc_auc"
    target_threshold: Optional[float] = None
    budget_trials: int = 30
    budget_seconds: Optional[int] = None
    cv_splits: int = 5
    random_state: int = 42
    enable_optuna: bool = True
    optuna_timeout_per_trial: Optional[int] = 60
    imbalance_threshold: float = 0.15
    hitl_enabled: bool = False
    hitl_auto_blocklist: List[str] = None

class ExperimentStore:
    def __init__(self, db_path: str = "agent_runs.sqlite"):
        self.db_path = db_path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                plan TEXT,
                result TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        con.commit()
        con.close()

    def log_trial(self, plan: TrialPlan, result: TrialResult):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO trials (ts, plan, result) VALUES (?, ?, ?)",
            (time.time(), json.dumps(asdict(plan)), json.dumps(asdict(result))),
        )
        con.commit()
        con.close()

    def save_artifact(self, key: str, value: Dict[str, Any]):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("REPLACE INTO artifacts (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        con.commit()
        con.close()

    def load_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("SELECT value FROM artifacts WHERE key=?", (key,))
        row = cur.fetchone()
        con.close()
        return json.loads(row[0]) if row else None

    def dataframe(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trials ORDER BY id ASC", con)
        con.close()
        if not df.empty:
            df["plan"] = df["plan"].apply(json.loads)
            df["result"] = df["result"].apply(json.loads)
        return df

class HumanInTheLoop:
    def __init__(self, enabled: bool = False, auto_blocklist: Optional[List[str]] = None):
        self.enabled = enabled
        self.auto_blocklist = auto_blocklist or []

    def approve_features(self, selected: List[str]) -> List[str]:
        if not self.enabled:
            return selected
        approved = []
        for f in selected:
            if any(b in f for b in self.auto_blocklist):
                continue
            approved.append(f)
        return approved

class LiteratureEnhancedAgent:
    def __init__(
        self,
        config: AgentConfig,
        pubmed_searcher: Optional[SimplePubMedSearcher] = None,
        store: Optional[ExperimentStore] = None,
        disease_context: Optional[str] = None,
        hitl: Optional[HumanInTheLoop] = None,
    ):
        self.cfg = config
        self.disease_context = disease_context
        self.pubmed_searcher = pubmed_searcher
        self.store = store or ExperimentStore()
        self.hitl = hitl or HumanInTheLoop(config.hitl_enabled, config.hitl_auto_blocklist)
        self.best_pipeline: Optional[Pipeline] = None
        self.best_score: float = -np.inf
        self.best_features: List[str] = []
        self.task_is_classification: Optional[bool] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.history: List[Tuple[TrialPlan, TrialResult]] = []
        self.literature_cache: Dict[str, dict] = {}

    def _sense(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        info["n_samples"] = len(X)
        info["n_features"] = X.shape[1]
        self.task_is_classification = self._is_classification(y)
        info["task"] = "classification" if self.task_is_classification else "regression"
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        info["n_numeric"] = len(self.numeric_cols)
        info["n_categorical"] = len(self.categorical_cols)
        if self.task_is_classification:
            vc = y.value_counts(normalize=True)
            min_class_ratio = float(vc.min())
            info["min_class_ratio"] = min_class_ratio
            info["imbalanced"] = min_class_ratio < self.cfg.imbalance_threshold
        else:
            info["y_skew"] = float(pd.Series(y).skew())
        try:
            if self.task_is_classification and set(pd.unique(y)) <= {0,1}:
                numeric = X[self.numeric_cols]
                if not numeric.empty:
                    corr = numeric.corrwith(y.astype(float)).abs()
                    info["max_abs_corr"] = float(corr.max())
                    info["leakage_suspect"] = bool(corr.max() > 0.98)
            else:
                numeric = X[self.numeric_cols]
                if not numeric.empty:
                    corr = numeric.corrwith(y).abs()
                    info["max_abs_corr"] = float(corr.max())
                    info["leakage_suspect"] = bool(corr.max() > 0.999)
        except Exception:
            info["max_abs_corr"] = np.nan
            info["leakage_suspect"] = False
        return info

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        unique = pd.unique(y)
        return (pd.api.types.is_integer_dtype(y) and len(unique) <= 20) or pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y)

    def _scorer_name(self) -> str:
        if self.cfg.target_metric:
            return self.cfg.target_metric
        return "f1_macro" if self.task_is_classification else "r2"

    def _cv(self):
        return StratifiedKFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state) if self.task_is_classification else KFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state)

    def _plan(self, sense_info: Dict[str, Any], prev_results: List[TrialResult]) -> TrialPlan:
        n, p = sense_info["n_samples"], sense_info["n_features"]
        imbalanced = sense_info.get("imbalanced", False)
        p_over_n = p / max(1, n)
        if p_over_n > 1.5:
            candidate_family = "l1"
        elif sense_info["n_categorical"] > sense_info["n_numeric"]:
            candidate_family = "mi"
        elif imbalanced and self.task_is_classification:
            candidate_family = "tree"
        else:
            candidate_family = "kbest"

        params = {
            "k": min( max(5, p // 4), max(1, p - 1) ),
            "step": 0.2,
            "C": 1.0,
            "alpha": 0.001,
            "n_estimators": 300,
        }

        comment = f"family={candidate_family}; p/n={p_over_n:.2f}; imbalanced={imbalanced}"
        return TrialPlan(strategy=candidate_family, params=params, comment=comment)

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", Pipeline([("sc", StandardScaler())]), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols))
        if not transformers:
            return ColumnTransformer([], remainder="passthrough")
        return ColumnTransformer(transformers, remainder="drop")

    def _make_selector(self, plan: TrialPlan, task_is_cls: bool) -> Tuple[str, Any]:
        k = plan.params.get("k", 10)
        if plan.strategy == "kbest":
            score_fn = f_classif if task_is_cls else f_regression
            return "sel", SelectKBest(score_fn, k=k)
        if plan.strategy == "mi":
            score_fn = mutual_info_classif if task_is_cls else mutual_info_regression
            return "sel", SelectKBest(score_fn, k=k)
        if plan.strategy == "rfe":
            base = LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if task_is_cls else LinearRegression()
            return "sel", RFE(estimator=base, n_features_to_select=k, step=plan.params.get("step", 0.2))
        if plan.strategy == "l1":
            if task_is_cls:
                model = LogisticRegression(penalty="l1", solver="saga", C=plan.params.get("C",1.0), max_iter=3000, random_state=self.cfg.random_state)
            else:
                model = Lasso(alpha=plan.params.get("alpha",0.001), max_iter=5000, random_state=self.cfg.random_state)
            return "sel", SelectFromModel(model, prefit=False)
        if plan.strategy == "tree":
            model = RandomForestClassifier(n_estimators=plan.params.get("n_estimators",300), random_state=self.cfg.random_state, n_jobs=-1) if task_is_cls else RandomForestRegressor(n_estimators=plan.params.get("n_estimators",300), random_state=self.cfg.random_state, n_jobs=-1)
            return "sel", SelectFromModel(model, prefit=False)
        if plan.strategy == "variance":
            return "sel", VarianceThreshold(threshold=1e-5)
        return "sel", VarianceThreshold(threshold=0.0)

    def _default_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if self.task_is_classification else LinearRegression()

    def _act_build_pipeline(self, plan: TrialPlan) -> Pipeline:
        pre = self._build_preprocessor()
        sel_name, selector = self._make_selector(plan, self.task_is_classification)
        model = self._default_estimator()
        pipe = Pipeline([
            ("prep", pre),
            (sel_name, selector),
            ("model", model),
        ])
        return pipe

    def _evaluate(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> TrialResult:
        metric = self._scorer_name()
        cv = self._cv()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric, n_jobs=-1)
        start_fit = time.time()
        pipe.fit(X, y)
        n_features = self._infer_selected_feature_count(pipe, X)
        selected = self._infer_selected_feature_names(pipe, X)
        duration = (time.time() - start_fit) + 0.0
        return TrialResult(
            metric_name=metric,
            metric_value=float(np.mean(scores)),
            metric_std=float(np.std(scores)),
            n_features=n_features,
            selected_features=selected,
            pipeline_repr=str(pipe),
            duration_sec=duration,
        )

    def _infer_selected_feature_count(self, pipe: Pipeline, X: pd.DataFrame) -> int:
        try:
            sel = pipe.named_steps.get("sel")
            if hasattr(sel, "get_support"):
                Xt = pipe.named_steps["prep"].fit_transform(X)
                return int(sel.fit(Xt, np.zeros(len(X))).get_support().sum())
        except Exception:
            pass
        return X.shape[1]

    def _infer_selected_feature_names(self, pipe: Pipeline, X: pd.DataFrame) -> List[str]:
        try:
            prep: ColumnTransformer = pipe.named_steps["prep"]
            sel = pipe.named_steps.get("sel")
            feature_names = []
            for name, trans, cols in prep.transformers_:
                if name == "remainder":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    outs = trans.get_feature_names_out(cols)
                else:
                    outs = cols
                feature_names.extend(list(outs))
            if hasattr(sel, "get_support"):
                Xt = prep.transform(X)
                sel.fit(Xt, np.zeros(len(X)))
                mask = sel.get_support()
                if mask is not None and len(feature_names) == len(mask):
                    return [f for f, m in zip(feature_names, mask) if m]
        except Exception:
            pass
        return list(X.columns)

    def _reflect(self, plan: TrialPlan, result: TrialResult, sense_info: Dict[str, Any]) -> TrialPlan:
        # Original reflection logic
        if len(self.history) >= 2:
            if result.metric_value < self.best_score + 1e-4:
                if plan.strategy in {"kbest", "mi"}:
                    plan.params["k"] = max(5, int(plan.params.get("k",10) * 1.5))
                    plan.comment += "; reflect: increase k"
                elif plan.strategy in {"l1","tree"}:
                    plan.strategy = "rfe" if plan.strategy == "l1" else "kbest"
                    plan.comment += "; reflect: switch family"
        
        if sense_info.get("imbalanced", False) and self.task_is_classification and self._scorer_name() not in {"roc_auc","average_precision"}:
            self.cfg.target_metric = "roc_auc"
            plan.comment += "; reflect: set metric=roc_auc"
        
        # Literature-informed reflection
        if self.pubmed_searcher and len(result.selected_features) <= 5:  # Limit for rate limiting
            try:
                lit_scores = []
                for feature in result.selected_features[:5]:
                    if feature not in self.literature_cache:
                        lit_result = self.pubmed_searcher.search_simple(feature,disease_context=self.disease_context, max_results=3)
                        self.literature_cache[feature] = lit_result
                    
                    lit_scores.append(self.literature_cache[feature]['evidence_score'])
                
                if lit_scores:
                    avg_lit_score = sum(lit_scores) / len(lit_scores)
                    
                    # Adjust strategy based on literature support
                    if avg_lit_score < 1.5:  # Low literature support
                        if plan.strategy == "kbest":
                            plan.params["k"] = max(5, int(plan.params["k"] * 0.8))
                            plan.comment += f"; lit_adj: reduce k (low_lit={avg_lit_score:.1f})"
                    elif avg_lit_score > 3.0:  # High literature support
                        plan.comment += f"; lit_adj: keep strategy (high_lit={avg_lit_score:.1f})"
                        
            except Exception as e:
                plan.comment += "; lit_adj: error"
        
        return plan

    def _stop_check(self, start_time: float, trials_done: int, last_result: Optional[TrialResult]) -> bool:
        if trials_done >= self.cfg.budget_trials:
            return True
        if self.cfg.budget_seconds is not None and (time.time() - start_time) >= self.cfg.budget_seconds:
            return True
        if self.cfg.target_threshold is not None and last_result is not None:
            if last_result.metric_value >= self.cfg.target_threshold:
                return True
        return False

    def _maybe_optuna_tune(self, plan: TrialPlan, X: pd.DataFrame, y: pd.Series) -> TrialPlan:
        if not self.cfg.enable_optuna:
            return plan
        try:
            import optuna
        except Exception:
            return plan
        # Simplified optuna integration for Streamlit
        return plan

    def run(self, X: pd.DataFrame, y: pd.Series, progress_callback=None) -> Dict[str, Any]:
        start = time.time()
        sense_info = self._sense(X, y)
        plan = self._plan(sense_info, [])

        trials = 0
        last_result: Optional[TrialResult] = None
        
        while True:
            plan = self._maybe_optuna_tune(plan, X, y)
            pipe = self._act_build_pipeline(plan)
            result = self._evaluate(pipe, X, y)

            approved = self.hitl.approve_features(result.selected_features)
            result.selected_features = approved

            self.store.log_trial(plan, result)
            self.history.append((plan, result))

            if result.metric_value > self.best_score:
                self.best_score = result.metric_value
                self.best_pipeline = pipe
                self.best_features = approved
                self.store.save_artifact(
                    "best",
                    {
                        "metric": result.metric_name,
                        "score": result.metric_value,
                        "n_features": result.n_features,
                        "features": approved,
                        "pipeline": result.pipeline_repr,
                        "plan": asdict(plan),
                    },
                )

            trials += 1
            last_result = result
            
            # Progress callback for Streamlit
            if progress_callback:
                progress_callback(trials, self.cfg.budget_trials, result)

            new_plan = self._reflect(plan, result, sense_info)
            plan = new_plan

            if self._stop_check(start, trials, last_result):
                break

        elapsed = time.time() - start
        return {
            "best_score": self.best_score,
            "best_metric": self._scorer_name(),
            "best_features": self.best_features,
            "trials": trials,
            "elapsed_sec": elapsed,
            "sense_info": sense_info,
            "history_df": self.store.dataframe(),
            "literature_cache": self.literature_cache if self.pubmed_searcher else {}
        }

# ---- Literature Analysis Functions ----
def create_literature_visualization(literature_results: List[dict]):
    """Create literature analysis visualizations"""
    if not literature_results:
        return None, None
    
    # Evidence scores bar chart
    df = pd.DataFrame(literature_results)
    df = df.sort_values('evidence_score', ascending=True)
    
    fig1 = px.bar(
        df.tail(15), 
        x='evidence_score', 
        y='feature_name',
        orientation='h',
        title="Literature Evidence Scores by Feature",
        labels={"evidence_score": "Evidence Score (0-5)", "feature_name": "Feature"},
        color='evidence_score',
        color_continuous_scale='Viridis'
    )
    fig1.update_layout(height=500)
    
    # Paper count distribution
    fig2 = px.histogram(
        df, 
        x='paper_count',
        nbins=20,
        title="Distribution of Paper Counts",
        labels={"paper_count": "Number of Papers", "count": "Number of Features"}
    )
    
    return fig1, fig2

def analyze_literature_results(literature_results: List[dict]) -> dict:
    """Analyze literature search results"""
    if not literature_results:
        return {}
    
    df = pd.DataFrame(literature_results)
    
    analysis = {
        'total_features': len(df),
        'total_papers': df['paper_count'].sum(),
        'avg_evidence_score': df['evidence_score'].mean(),
        'high_evidence_features': len(df[df['evidence_score'] > 2.0]),
        'zero_evidence_features': len(df[df['paper_count'] == 0]),
        'top_features': df.nlargest(5, 'evidence_score')[['feature_name', 'evidence_score', 'paper_count']].to_dict('records')
    }
    
    return analysis

def display_articles_for_feature(feature_name: str, articles: List[dict]):
    """Display articles for a specific feature in Streamlit"""
    if not articles:
        st.info(f"No articles found for {feature_name}")
        return
    
    st.markdown(f"**📚 Publications for {feature_name}:**")
    
    for i, article in enumerate(articles, 1):
        with st.expander(f"{i}. {article['title'][:100]}..."):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Authors:** {article['authors']}")
                st.markdown(f"**Journal:** {article['journal']}")
                st.markdown(f"**Year:** {article['year']}")
                if article['abstract'] != "No abstract available":
                    st.markdown(f"**Abstract:** {article['abstract']}")
            with col2:
                st.markdown(f"**PMID:** {article['pmid']}")
                st.markdown(f"[View on PubMed]({article['url']})")

# ---- Streamlit Application ----

def main():
    st.set_page_config(
        page_title="🤖 PubMed-Enhanced Feature Selection Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 PubMed-Enhanced Feature Selection Agent")
    st.markdown("""
    **Autonomous feature selection agent with literature analysis** — automatically picks the best features and validates them against scientific literature!
    
    📊 **New Features:**
    - 🔬 **PubMed Integration**: Automatic literature search for selected features
    - 📚 **Evidence Scoring**: Features ranked by scientific publication support
    - 🎯 **Literature-Informed Decisions**: Agent adapts strategy based on literature evidence
    - 📈 **Publication Analytics**: Visualize research trends and evidence strength
    - 📖 **Article Listing**: View actual PubMed articles for each selected feature
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Dataset for feature selection"
    )

    # PubMed Configuration
    st.sidebar.subheader("🔬 PubMed Literature Analysis")
    enable_pubmed = st.sidebar.checkbox("Enable PubMed Search", value=False, help="Search scientific literature for selected features")
    
    pubmed_searcher = None
    if enable_pubmed:
        email = st.sidebar.text_input(
            "NCBI Email (required)", 
            help="Required by NCBI for PubMed API access - get free account at ncbi.nlm.nih.gov"
        )
        
        api_key = st.sidebar.text_input(
            "NCBI API Key (optional)",
            type="password",
            help="Optional - increases rate limit from 3 to 10 requests/sec"
        )
        
        disease_context = st.sidebar.text_input(
            "Disease/Condition Context",
            placeholder="e.g., cancer, diabetes, alzheimer",
            help="Helps focus literature search on specific medical condition"
        )
        
        if email:
            pubmed_searcher = SimplePubMedSearcher(
                email=email, 
                api_key=api_key if api_key else None
            )
            st.sidebar.success("✅ PubMed search enabled")
        else:
            st.sidebar.warning("⚠️ Email required for PubMed API")
            enable_pubmed = False

    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"❌ File upload error: {str(e)}")
            return

        # Target column selection
        target_col = st.sidebar.selectbox(
            "🎯 Select target variable",
            options=df.columns.tolist(),
            index=len(df.columns)-1,
            help="Variable to predict"
        )

        # Configuration
        st.sidebar.subheader("🔧 Agent Settings")
        
        # Task type detection
        y = df[target_col]
        is_classification = len(y.unique()) <= 20 and (y.dtype == 'object' or y.dtype == 'int64')
        task_type = "Classification" if is_classification else "Regression"
        st.sidebar.info(f"📋 Detected task: **{task_type}**")

        # Metric selection
        if is_classification:
            default_metrics = ["roc_auc", "f1_macro", "accuracy", "precision_macro", "recall_macro"]
        else:
            default_metrics = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        
        target_metric = st.sidebar.selectbox(
            "📊 Target metric",
            options=default_metrics,
            help="Metric to optimize"
        )

        # Budget settings
        budget_trials = st.sidebar.slider(
            "🔄 Max trials",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of trials to run"
        )

        budget_seconds = st.sidebar.slider(
            "⏱️ Max time (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            help="Maximum runtime"
        )

        cv_splits = st.sidebar.slider(
            "🔀 Number of CV folds",
            min_value=3,
            max_value=10,
            value=5
        )

        # Advanced settings
        with st.sidebar.expander("🔬 Advanced Settings"):
            target_threshold = st.number_input(
                "Target threshold (optional)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Stop early if reached (0 = disabled)"
            )
            
            enable_optuna = st.checkbox(
                "Optuna hyperparameter optimization",
                value=False,
                help="Better results but slower"
            )
            
            imbalance_threshold = st.slider(
                "Class imbalance threshold",
                min_value=0.05,
                max_value=0.5,
                value=0.15,
                help="Minimum class ratio"
            )

            hitl_enabled = st.checkbox(
                "Human-in-the-loop approval",
                value=False,
                help="Manually approve selected features"
            )

            if hitl_enabled:
                blocklist_text = st.text_input(
                    "Blocked features (comma-separated)",
                    help="Features containing these names will be auto-rejected"
                )
                hitl_auto_blocklist = [x.strip() for x in blocklist_text.split(",") if x.strip()]
            else:
                hitl_auto_blocklist = []

        # Run button
        if st.sidebar.button("🚀 Start Feature Selection", type="primary"):
            # Prepare data
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Create config
            config = AgentConfig(
                target_metric=target_metric,
                target_threshold=target_threshold if target_threshold > 0 else None,
                budget_trials=budget_trials,
                budget_seconds=budget_seconds,
                cv_splits=cv_splits,
                random_state=42,
                enable_optuna=enable_optuna,
                imbalance_threshold=imbalance_threshold,
                hitl_enabled=hitl_enabled,
                hitl_auto_blocklist=hitl_auto_blocklist
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()

            # Results containers
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Real-time Progress")
                progress_chart = st.empty()
                
            with col2:
                st.subheader("📈 Best Results")
                best_metrics = st.empty()

            # Initialize tracking
            trial_scores = []
            trial_features = []

            def progress_callback(trial_num, total_trials, result):
                progress = trial_num / total_trials
                progress_bar.progress(progress)
                status_text.text(f"Trial {trial_num}/{total_trials} - Current score: {result.metric_value:.4f}")
                
                # Track results
                trial_scores.append(result.metric_value)
                trial_features.append(result.n_features)
                
                # Update progress chart
                if len(trial_scores) > 1:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=["Metric Value", "Number of Features"],
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            y=trial_scores,
                            mode='lines+markers',
                            name='Metric',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            y=trial_features,
                            mode='lines+markers',
                            name='Features',
                            line=dict(color='green')
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    progress_chart.plotly_chart(fig, use_container_width=True)

                # Update best metrics
                best_score = max(trial_scores)
                best_idx = trial_scores.index(best_score)
                best_n_features = trial_features[best_idx]
                
                best_metrics.metric(
                    label=f"Best {target_metric.upper()}",
                    value=f"{best_score:.4f}",
                    delta=f"{best_n_features} features"
                )

            # Run agent
            try:
                agent = LiteratureEnhancedAgent(
                    config, 
                    pubmed_searcher, 
                    disease_context=disease_context if disease_context else None
                )
                
                with st.spinner("🤖 Running feature selection agent..."):
                    results = agent.run(X, y, progress_callback=progress_callback)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display final results
                st.success("✅ Feature selection completed!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Score", f"{results['best_score']:.4f}")
                with col2:
                    st.metric("Selected Features", len(results['best_features']))
                with col3:
                    st.metric("Total Trials", results['trials'])
                with col4:
                    st.metric("Duration", f"{results['elapsed_sec']:.1f}s")

                # Data analysis summary
                st.subheader("🔍 Data Analysis Summary")
                sense_info = results['sense_info']
                
                analysis_col1, analysis_col2 = st.columns(2)
                with analysis_col1:
                    st.info(f"""
                    **Dataset Info:**
                    - Samples: {sense_info['n_samples']:,}
                    - Total features: {sense_info['n_features']}
                    - Numeric features: {sense_info['n_numeric']}
                    - Categorical features: {sense_info['n_categorical']}
                    - Task type: {sense_info['task']}
                    """)
                
                with analysis_col2:
                    warnings_list = []
                    if sense_info.get('imbalanced', False):
                        warnings_list.append(f"⚠️ Class imbalance detected (min ratio: {sense_info.get('min_class_ratio', 0):.3f})")
                    if sense_info.get('leakage_suspect', False):
                        warnings_list.append(f"🚨 Possible data leakage (max corr: {sense_info.get('max_abs_corr', 0):.3f})")
                    
                    if warnings_list:
                        st.warning("\n".join(warnings_list))
                    else:
                        st.success("✅ No data quality warnings")

                # PubMed Literature Analysis
                if enable_pubmed and pubmed_searcher and results['best_features']:
                    st.subheader("🔬 Literature Analysis")
                    
                    with st.spinner("🔍 Searching scientific literature..."):
                        # Create progress tracking for literature search
                        lit_progress = st.progress(0)
                        lit_status = st.empty()
                        
                        def lit_progress_callback(current, total, feature):
                            lit_progress.progress(current / total)
                            lit_status.text(f"Searching literature for: {feature} ({current}/{total})")
                        
                        # Perform batch literature search
                        literature_results = pubmed_searcher.batch_search(
                            results['best_features'][:15],  # Limit to avoid rate limits
                            disease_context if disease_context else None,
                            progress_callback=lit_progress_callback
                        )
                        
                        lit_progress.empty()
                        lit_status.empty()
                    
                    if literature_results:
                        # Literature analysis
                        lit_analysis = analyze_literature_results(literature_results)
                        
                        # Literature summary metrics
                        st.subheader("📚 Literature Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Papers Found", lit_analysis.get('total_papers', 0))
                        with col2:
                            st.metric("Avg Evidence Score", f"{lit_analysis.get('avg_evidence_score', 0):.1f}/5.0")
                        with col3:
                            st.metric("High Evidence Features", lit_analysis.get('high_evidence_features', 0))
                        with col4:
                            st.metric("Zero Evidence Features", lit_analysis.get('zero_evidence_features', 0))
                        
                        # Literature visualizations
                        fig1, fig2 = create_literature_visualization(literature_results)
                        if fig1:
                            st.plotly_chart(fig1, use_container_width=True)
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Top features by literature evidence
                        if lit_analysis.get('top_features'):
                            st.subheader("🏆 Top Literature-Supported Features")
                            top_features_df = pd.DataFrame(lit_analysis['top_features'])
                            top_features_df['Evidence Score'] = top_features_df['evidence_score'].round(1)
                            top_features_df['Papers'] = top_features_df['paper_count']
                            st.dataframe(top_features_df[['feature_name', 'Evidence Score', 'Papers']], use_container_width=True)
                        
                        # *** NEW: Detailed Article Listings for Each Feature ***
                        st.subheader("📖 Detailed Article Listings")
                        
                        # Create tabs for each feature with articles
                        features_with_articles = [r for r in literature_results if r.get('articles', [])]
                        
                        if features_with_articles:
                            # Limit to top 10 features to avoid too many tabs
                            features_to_show = sorted(features_with_articles, 
                                                    key=lambda x: x['evidence_score'], 
                                                    reverse=True)[:10]
                            
                            # Create tabs for each feature
                            tab_labels = []
                            tab_contents = []
                            
                            for feature_data in features_to_show:
                                feature_name = feature_data['feature_name']
                                articles = feature_data.get('articles', [])
                                evidence_score = feature_data['evidence_score']
                                
                                # Create short tab label
                                short_name = feature_name[:15] + "..." if len(feature_name) > 15 else feature_name
                                tab_label = f"{short_name} ({evidence_score:.1f})"
                                tab_labels.append(tab_label)
                                
                                # Store content for later
                                tab_contents.append((feature_name, articles, feature_data))
                            
                            # Create tabs
                            if tab_labels:
                                tabs = st.tabs(tab_labels)
                                
                                for tab, (feature_name, articles, feature_data) in zip(tabs, tab_contents):
                                    with tab:
                                        # Feature header with stats
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Evidence Score", f"{feature_data['evidence_score']:.1f}/5.0")
                                        with col2:
                                            st.metric("Total Papers", feature_data['paper_count'])
                                        with col3:
                                            st.metric("Articles Fetched", len(articles))
                                        
                                        # Search query used
                                        if feature_data.get('search_query'):
                                            st.info(f"**Search Query Used:** {feature_data['search_query']}")
                                        
                                        # Display articles
                                        display_articles_for_feature(feature_name, articles)
                        else:
                            st.info("No articles were retrieved for the selected features. This could be due to:")
                            st.markdown("- Features not being found in biomedical literature")
                            st.markdown("- Search queries too specific")
                            st.markdown("- Rate limiting preventing article fetch")
                        
                        # Detailed literature table
                        with st.expander("📋 Detailed Literature Results"):
                            lit_df = pd.DataFrame([
                                {
                                    'Feature': r['feature_name'],
                                    'Papers': r['paper_count'],
                                    'Evidence Score': f"{r['evidence_score']:.1f}/5.0",
                                    'Articles Retrieved': len(r.get('articles', [])),
                                    'Support Level': '🔥 High' if r['evidence_score'] > 3.0 
                                                   else '📈 Medium' if r['evidence_score'] > 1.0 
                                                   else '❓ Low',
                                    'Search Query': r.get('search_query', '')[:50] + '...' if len(r.get('search_query', '')) > 50 else r.get('search_query', '')
                                }
                                for r in literature_results
                            ])
                            st.dataframe(lit_df, use_container_width=True)
                        
                        # Literature insights
                        insights = []
                        if lit_analysis['high_evidence_features'] > 0:
                            insights.append(f"🎯 {lit_analysis['high_evidence_features']} features have strong literature support")
                        if lit_analysis['total_papers'] > 50:
                            insights.append(f"📊 Extensive literature base with {lit_analysis['total_papers']} total papers")
                        if lit_analysis['zero_evidence_features'] > lit_analysis['total_features'] * 0.5:
                            insights.append("⚠️ Many features lack literature support - consider domain expert review")
                        
                        # Count features with articles
                        features_with_articles_count = len([r for r in literature_results if r.get('articles', [])])
                        if features_with_articles_count > 0:
                            insights.append(f"📚 Retrieved detailed articles for {features_with_articles_count} features")
                        
                        if insights:
                            st.success("**Literature Insights:**\n" + "\n".join(f"- {insight}" for insight in insights))

                # Selected features
                st.subheader("🏆 Selected Features")
                if results['best_features']:
                    features_df = pd.DataFrame({
                        'Feature': results['best_features'],
                        'Rank': range(1, len(results['best_features']) + 1)
                    })
                    
                    # Add literature scores if available
                    if enable_pubmed and 'literature_results' in locals():
                        lit_dict = {r['feature_name']: r['evidence_score'] for r in literature_results}
                        features_df['Literature Score'] = features_df['Feature'].map(lit_dict).fillna(0).round(1)
                    
                    st.dataframe(features_df, use_container_width=True)

                # Trial history
                st.subheader("📜 Trial History")
                if not results['history_df'].empty:
                    history_df = results['history_df']
                    
                    # Extract key metrics from nested JSON
                    history_display = []
                    for _, row in history_df.iterrows():
                        plan = row['plan']
                        result = row['result']
                        history_display.append({
                            'Trial': row['id'],
                            'Strategy': plan['strategy'],
                            'Metric': f"{result['metric_value']:.4f}",
                            '# Features': result['n_features'],
                            'Duration (s)': f"{result['duration_sec']:.2f}",
                            'Comment': plan.get('comment', '')[:50] + '...' if len(plan.get('comment', '')) > 50 else plan.get('comment', '')
                        })
                    
                    st.dataframe(pd.DataFrame(history_display), use_container_width=True)

                # Download section
                st.subheader("💾 Download Options")
                
                download_col1, download_col2, download_col3 = st.columns(3)
                
                with download_col1:
                    # Selected features CSV
                    features_download_df = pd.DataFrame({'selected_features': results['best_features']})
                    if enable_pubmed and 'literature_results' in locals():
                        lit_dict = {r['feature_name']: r for r in literature_results}
                        features_download_df['literature_score'] = features_download_df['selected_features'].map(
                            lambda x: lit_dict.get(x, {}).get('evidence_score', 0)
                        )
                        features_download_df['paper_count'] = features_download_df['selected_features'].map(
                            lambda x: lit_dict.get(x, {}).get('paper_count', 0)
                        )
                        features_download_df['articles_retrieved'] = features_download_df['selected_features'].map(
                            lambda x: len(lit_dict.get(x, {}).get('articles', []))
                        )
                    
                    features_csv = features_download_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Selected Features (CSV)",
                        data=features_csv,
                        file_name="selected_features_with_literature.csv",
                        mime="text/csv"
                    )
                
                with download_col2:
                    # Full results JSON
                    download_results = {
                        'best_score': results['best_score'],
                        'best_features': results['best_features'],
                        'config': asdict(config),
                        'sense_info': sense_info,
                        'literature_results': literature_results if enable_pubmed and 'literature_results' in locals() else []
                    }
                    results_json = json.dumps(download_results, indent=2)
                    st.download_button(
                        label="📊 Download Full Results (JSON)",
                        data=results_json,
                        file_name="feature_selection_results.json",
                        mime="application/json"
                    )
                
                with download_col3:
                    if enable_pubmed and 'literature_results' in locals():
                        # Literature report with articles
                        lit_report = {
                            'analysis': lit_analysis,
                            'detailed_results': literature_results,
                            'search_parameters': {
                                'disease_context': disease_context,
                                'email': email,
                                'api_key_used': bool(api_key)
                            }
                        }
                        lit_json = json.dumps(lit_report, indent=2)
                        st.download_button(
                            label="🔬 Download Literature Report (JSON)",
                            data=lit_json,
                            file_name="literature_analysis_report.json",
                            mime="application/json"
                        )

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

    else:
        # Welcome screen
        st.info("""
        👈 **To get started:**
        1. Upload your CSV from the left panel
        2. Select the target variable
        3. Enable PubMed search and enter your email 
        4. Enter API key for faster and better searches      
            i-go to https://ncbi.nlm.nih.gov and register for a free 
            account if you don't have one
                
            ii- get an API key from your account settings (optional)        
        5. Configure the settings
        6. Click the "Start Feature Selection" button
        
        📝 **Supported formats:**
        - CSV files
        - Both classification and regression tasks
        - Numeric and categorical features
        
        🔬 **PubMed Integration:**
        - Automatic literature search for selected features
        - Evidence scoring based on publication count
        - Literature-informed agent decisions
        - Detailed publication analytics
        - **NEW**: Full article listings with abstracts
        """)
        
        # Sample data option
        st.subheader("🎲 Try with Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏥 Load Breast Cancer Dataset"):
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer(as_frame=True)
                sample_df = data.frame
                
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Breast Cancer Sample",
                    data=csv,
                    file_name="sample_breast_cancer.csv",
                    mime="text/csv"
                )
                st.success("✅ Sample data ready for download!")
        
        with col2:
            if st.button("❤️ Load Heart Disease Dataset"):
                try:
                    # Create a synthetic heart disease dataset

                    heart_df = generate_heart_df(n_samples=2000, seed=7)
                    
                    csv = heart_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Heart Disease Sample",
                        data=csv,
                        file_name="sample_heart_disease.csv",
                        mime="text/csv"
                    )
                    st.success("✅ Sample data ready for download!")
                except Exception as e:
                    st.error(f"Error creating sample: {e}")

        # Info about PubMed setup
        st.subheader("🔬 Setting up PubMed Search")
        st.info("""
        **To use the PubMed literature analysis feature:**
        
        1. **Get a free NCBI account**: Visit [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/) and register
        2. **Use your email**: Enter the email associated with your NCBI account
        3. **Optional API Key**: Get an API key from your NCBI account settings for faster searches (10 req/sec vs 3 req/sec)
        4. **Disease Context**: Specify a medical condition to focus the literature search (e.g., "cancer", "diabetes")
        
        **What you get:**
        - Automatic PubMed search for each selected feature
        - Evidence scores based on publication count and relevance
        - Literature-informed agent decisions
        - **NEW**: Detailed article listings with titles, authors, abstracts, and PubMed links
        - Downloadable publication analysis reports
        """)

if __name__ == "__main__":
    main()
