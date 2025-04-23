import pandas as pd
import re
import tldextract
from rapidfuzz import fuzz
import phonenumbers
from phonenumbers import geocoder

# Load the dataset from a Parquet file
df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet", engine='pyarrow')

# Normalization functions
def normalize_domain(url):
    if pd.isna(url):
        return ""
    ext = tldextract.extract(url)
    return ext.domain if ext.domain else ""

def normalize_company_name(text):
    if pd.isna(text):
        return ""
    
    legal_terms = [
        r"\bSRL\b", r"\bS\.R\.L\.?\b", r"\bLLC\b", r"\bL\.L\.C\.?\b",
        r"\bPVT\b", r"\bP\.V\.T\.?\b", r"\bINC\b", r"\bINC\.?\b",
        r"\bSA\b", r"\bS\.A\.?\b", r"\bSC\b", r"\bS\.C\.?\b",
        r"\bGMBH\b", r"\bG\.M\.B\.H\.?\b", r"\bLTD\b", r"\bL\.T\.D\.?\b",
        r"\bLIMITED\b", r"\bCORP\b", r"\bCORPORATION\b", r"\bPLC\b",
        r"\bN\.V\.?\b", r"\bAG\b", r"\bOY\b", r"\bAB\b",
        r"\bBV\b", r"\bAS\b", r"\bSAS\b", r"\bS\.A\.S\.?\b",
        r"\bS\.P\.A\.?\b", r"\bK\.K\.?\b", r"\bL\.L\.P\.?\b", r"\bLLP\b",
        r"\bCO\b", r"\bCO\.?\b", r"\bORG\b", r"\bORG\.?\b"
    ]

    text = text.split('|')[0]  # Remove after |
    pattern = re.compile('|'.join(legal_terms), re.IGNORECASE)
    text = pattern.sub('', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text[:50] if text else ""

def normalize_country_code(row):
    phone_number = row['primary_phone']
    country = row['main_country_code']
    if pd.isna(country):
        return get_country_code_by_phone(phone_number)
    country = country.strip().upper()

    return country

def normalize_primary_phone(phone_number):
    cleaned = re.sub(r"\D", "", phone_number)
    return str(cleaned)  

def normalize_email(email):
    if not email:
        return ""
    
    email = email.lower()

    if "@gmail.com" in email:
        local, domain = email.split('@')
        local = local.replace('.', '') 
        email = local + '@' + domain
    
    email = re.sub(r'[^a-z0-9@._-]', '', email)
    
    return email
    
def normalize_facebook_url(url):
    if not isinstance(url, str) or 'facebook.com' not in url:
        return ""

    url = url.strip().lower()

    url = url.split('?')[0].split('#')[0]

    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "").replace("m.", "")

    parts = url.split("facebook.com/")
    if len(parts) < 2:
        return ""

    username = parts[1].strip("/")

    if username.startswith("profile.php"):
        return ""

    return username

def get_country_code_by_phone(phone_number):
    try:
        parsed_number = phonenumbers.parse(phone_number)
        
        country_code = geocoder.region_code_for_number(parsed_number)
        
        if country_code:
            return country_code
        else:
            return "unknown"
    
    except phonenumbers.phonenumberutil.NumberParseException:
        return "unknown"

# Function to group companies based on normalized fields and similarity
def group_companies():
    # Initialize the group_id column
    df["group_id"] = -1  
    group_id_counter = 0  

    # Group similar companies within each country 
    for country_code_normalized in df["country_code_normalized"].unique():
        
        subset = df[df["country_code_normalized"] == country_code_normalized]
        seen = {}  # Stores previously seen companies and their metadata
        # seen = { company_name: (group_id, domain, phone, facebook_url) }
        # seen = {
        #     microsoft: (0, "microsoft", "+34625223431", "microsoftofficialpage"),
        #     tesla: (1, "tesla", "+34625223431", "teslafficialpage",
        #     google: (2, "google", "+34625223431", "googlefficialpage")
        # }

        for idx, name in subset["company_name_normalized"].items():
            matched = False
            domain = df.at[idx, "website_domain_normalized"]
            phone = df.at[idx, "primary_phone"]
            facebook_URL = df.at[idx, "facebook_url"]

            # Compare current company to all previously seen in this country
            for seen_name, (group_id, seen_domain, seen_phone, seen_facebook_URL) in seen.items():
                score = fuzz.token_sort_ratio(name, seen_name)

                if score > 90:
                    # Strong match regardless of domain
                    df.at[idx, "group_id"] = group_id
                    matched = True
                    break
                elif 70 < score <= 90 and domain and domain == seen_domain:
                    # Match only if domain is same in mid similarity range
                    df.at[idx, "group_id"] = group_id
                    matched = True
                    break
                elif 30 < score <= 70 and ((phone and phone == seen_phone) or (facebook_URL and facebook_URL == seen_facebook_URL)) :
                    # Match only if domain is same in mid similarity range
                    df.at[idx, "group_id"] = group_id
                    matched = True
                    break
            if not matched:
                # New group
                seen[name] = (group_id_counter, domain, phone, facebook_URL)
                df.at[idx, "group_id"] = group_id_counter
                group_id_counter += 1

# Main pipeline execution
def run():
    # Apply normalization and insert column
    df.insert(0, "website_domain_normalized", df["website_domain"].apply(normalize_domain))
    df.insert(1, "company_name_normalized", df["company_name"].apply(normalize_company_name))
    df.insert(2, "country_code_normalized", df.apply(normalize_country_code, axis=1))
    df.insert(3, "primary_email_normalized", df["primary_email"].apply(normalize_email))
    df.insert(4, "facebook_url_normalized", df["facebook_url"].apply(normalize_facebook_url))

    # Group similar entities
    group_companies()

    # Move group_id column to the front
    df.insert(0, 'group_id', df.pop('group_id'))

    # Save results to a new parquet file
    df.to_parquet('grouped_entities.parquet', engine='pyarrow', index=False)

# Run the full process
run()