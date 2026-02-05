import pandas as pd
from typing import Dict, List, Tuple
import re

class ProductRecommender:
    def __init__(self, products_path: str):
        self.products_df = self._load_and_clean_data(products_path)
    
    def _load_and_clean_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()  # Normalize column names
        return df
    
    def get_recommendations(
        self,
        skin_data: Dict,       # From skin detection
        user_answers: Dict     # From questions page
    ) -> Dict[str, List[Dict]]:
        skin_types = self._determine_skin_types(
            skin_data['primary_type'],
            skin_data.get('concerns', [])
        )
        
        filtered = self._filter_products(
            skin_types,
            user_answers['age'],
            user_answers['budget']
        )
        
        return self._organize_recommendations(filtered)
    
    def _determine_skin_types(self, primary_type: str, concerns: List[Tuple[str, float]]) -> List[str]:
        primary_type = primary_type.lower().replace(' ', '_')
        types = [primary_type]
        
        if primary_type == 'combination':
            return ['oily', 'normal_to_dry']
        
        for concern, prob in concerns:
            if ('wrinkle' in concern.lower() or 'fine_line' in concern.lower()) and prob > 0.4:
                types.append('fine_lines_wrinkles')
                break
        
        return types
    
    def _filter_products(self, skin_types: List[str], age_group: str, budget: str) -> pd.DataFrame:
        df = self.products_df
        filtered = df[df['skin type'].str.contains('|'.join(skin_types), case=False)]

        # Age group filtering
        if age_group == '30 and above':
            filtered = filtered[(filtered['min age'] >= 30) | (filtered['category'].str.lower() == 'serum')]
        elif age_group == '20s':
            filtered = filtered[filtered['min age'] <= 29]
        else:  # '19 and younger'
            filtered = filtered[filtered['min age'] <= 19]
        
        # Budget filtering
        if budget == 'below 500':
            filtered = filtered[filtered['price'] <= 500]
        elif budget == '500-1000':
            filtered = filtered[(filtered['price'] > 500) & (filtered['price'] <= 1000)]
        elif budget == '1000+':
            filtered = filtered[filtered['price'] > 1000]

        return filtered

    def _organize_recommendations(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        categories = ['cleanser', 'toner', 'serum', 'moisturiser', 'moisturizer', 'sunscreen', 'face mask']
        results = {cat: [] for cat in categories}

        for _, row in df.iterrows():
            cat = row['category'].lower()
            if cat in results:
                # Use img column directly (just the file name)
                image_filename = row['img'].split('/')[-1] if pd.notna(row['img']) else ''
                image_path = f"images/products/{image_filename}" if image_filename else ''

                results[cat].append({
                    'name': row['product name'],
                    'brand': row['brand'],
                    'price': row['price'],
                    'image': image_path,
                    'description': row['description']
                })

        # Sort by price descending
        for cat in results:
            results[cat].sort(key=lambda x: x['price'], reverse=True)

        return results
