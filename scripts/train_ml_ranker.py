import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.storage import MetadataStorage
from app.services.jd_parser import JDParserService
from app.services.embedding_service import EmbeddingService
from app.services.bm25_search import BM25SearchService
from app.services.skill_matcher import SkillMatcherService
from app.services.experience_matcher import ExperienceMatcherService
from app.services.education_matcher import EducationMatcherService

def train_ranker():
    print("=" * 65)
    print("      TRAINING OPTIONAL MACHINE LEARNING RANKER (XGBOOST)")
    print("=" * 65)

    storage = MetadataStorage()
    jd_parser = JDParserService()
    embedding_service = EmbeddingService()
    bm25_service = BM25SearchService()
    skill_service = SkillMatcherService()
    exp_service = ExperienceMatcherService()
    edu_service = EducationMatcherService()

    resumes = storage.get_all_resumes()
    if len(resumes) < 10:
        print("Need at least 10 resumes indexed.")
        return

    sample_jds = [
        "Senior NLP Machine Learning Engineer with Python PyTorch Transformers BERT 3+ years experience",
        "Full Stack Python Developer with FastAPI Django React PostgreSQL Docker 4+ years experience",
        "DevOps Cloud Engineer with AWS Kubernetes Docker Terraform CI/CD Linux 2+ years experience",
        "Data Scientist with Python SQL Machine Learning Pandas Scikit-Learn 2+ years experience"
    ]

    features_list = []
    labels_list = []

    print("Generating feature vectors across JD/Resume pairs...")
    for jd_text in sample_jds:
        parsed_jd = jd_parser.parse_job_description(jd_text)
        jd_vecs = embedding_service.compute_jd_vectors(parsed_jd)
        bm25_scores = bm25_service.search(parsed_jd.raw_text, top_k=len(resumes))

        for r in resumes:
            cand_vecs = embedding_service.compute_resume_vectors(r)
            
            # Feature extraction
            sem_sim = float(np.dot(cand_vecs["full"], jd_vecs["full"]))
            proj_sim = float(np.dot(cand_vecs["projects"], jd_vecs["projects"]))
            bm25_val = bm25_scores.get(r.candidate_id, 0.0)
            
            skill_score, matched_req, missing_req, matched_pref = skill_service.evaluate_skills(r, parsed_jd)
            req_skill_ratio = len(matched_req) / len(parsed_jd.required_skills) if parsed_jd.required_skills else 1.0
            pref_skill_ratio = len(matched_pref) / len(parsed_jd.preferred_skills) if parsed_jd.preferred_skills else 0.5
            
            exp_score = exp_service.evaluate_experience(r, parsed_jd)
            edu_score, _ = edu_service.evaluate_education(r, parsed_jd)

            # Ground truth relevance label
            cand_skills_set = set(r.skills)
            jd_req_set = set(parsed_jd.required_skills)
            overlap = len(cand_skills_set & jd_req_set)
            
            if overlap >= 3 and exp_score >= 0.7:
                label = 2  # Highly Relevant
            elif overlap >= 1:
                label = 1  # Partially Relevant
            else:
                label = 0  # Not Relevant

            feat = [
                sem_sim,
                bm25_val,
                skill_score,
                req_skill_ratio,
                pref_skill_ratio,
                exp_score,
                edu_score,
                proj_sim
            ]
            features_list.append(feat)
            labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"Dataset generated: {len(X)} samples with {X.shape[1]} features.")
    print("Class distribution:", pd.Series(y).value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Not Relevant (0)", "Partially Relevant (1)", "Highly Relevant (2)"], zero_division=0))

    # Feature Importance
    feat_names = [
        "semantic_similarity", "bm25_score", "skill_score",
        "req_skill_match", "pref_skill_match", "experience_match",
        "education_match", "project_similarity"
    ]
    importances = model.feature_importances_
    print("Feature Importances:")
    for fn, imp in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {fn:<22}: {imp:.4f}")

    # Save model
    model_path = Path(__file__).resolve().parent.parent / "data" / "xgb_ranker.joblib"
    joblib.dump(model, model_path)
    print(f"\nTrained ranking model successfully saved to: {model_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_ranker()
