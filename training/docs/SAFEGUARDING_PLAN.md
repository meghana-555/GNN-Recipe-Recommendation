# Safeguarding Plan for Mealie Recipe Recommendation System

## Overview

This document outlines the safeguarding measures implemented in the Mealie Recipe Recommendation ML system to ensure fairness, explainability, transparency, privacy, accountability, and robustness.

---

## 1. Fairness

### 1.1 Data Fairness
- **Bias Detection in Training Data**: Before each training run, we analyze the distribution of:
  - Ratings across different cuisine types
  - User activity levels
  - Recipe popularity (long-tail analysis)
  
- **Mitigation Strategy**: 
  - Apply sample weighting to prevent popular recipes from dominating recommendations
  - Ensure minimum representation of diverse recipe categories
  - Monitor for demographic disparities in recommendation patterns

### 1.2 Recommendation Fairness
- **Exposure Fairness**: Track recommendation exposure across recipe categories
- **Novelty-Popularity Balance**: Ensure recommendations include both popular and niche recipes
- **Cold Start Handling**: Fair treatment for new users and new recipes

### 1.3 Implementation
```python
# In train.py - Add fairness metrics logging
def compute_fairness_metrics(interactions_df, recipes_df):
    """Compute fairness metrics for training data."""
    # Category distribution
    category_counts = recipes_df.groupby('category').size()
    gini_coefficient = compute_gini(category_counts)
    
    # User activity distribution
    user_ratings = interactions_df.groupby('u').size()
    user_gini = compute_gini(user_ratings)
    
    return {
        'category_gini': gini_coefficient,
        'user_activity_gini': user_gini,
        'num_cold_start_users': (user_ratings < 5).sum(),
    }
```

---

## 2. Explainability

### 2.1 Model Explainability
- **Feature Attribution**: Use GNN attention weights to identify important features
- **Recommendation Explanations**: Generate human-readable explanations like:
  - "Recommended because you liked similar Italian pasta dishes"
  - "Popular among users with similar taste preferences"
  - "Contains ingredients you frequently use"

### 2.2 Implementation
```python
# In inference service
def get_recommendation_explanation(user_id, recipe_id, model, data):
    """Generate human-readable explanation for a recommendation."""
    # Get user and recipe features
    user_features = get_user_features(user_id, data)
    recipe_features = get_recipe_features(recipe_id, data)
    
    explanations = []
    
    # Check ingredient overlap
    common_ingredients = set(user_features['frequent_ingredients']) & set(recipe_features['ingredients'])
    if common_ingredients:
        explanations.append(f"Uses ingredients you frequently cook with: {', '.join(list(common_ingredients)[:3])}")
    
    # Check cuisine preference
    if recipe_features['cuisine'] in user_features['preferred_cuisines']:
        explanations.append(f"Matches your preference for {recipe_features['cuisine']} cuisine")
    
    # Check similar user behavior
    similar_users_who_liked = get_similar_users_who_rated(user_id, recipe_id)
    if similar_users_who_liked > 5:
        explanations.append("Highly rated by users with similar tastes")
    
    return explanations
```

### 2.3 Logging & Audit Trail
- All recommendations are logged with:
  - Model version used
  - Input features
  - Output scores
  - Explanation factors

---

## 3. Transparency

### 3.1 Model Card
We maintain a model card (see `docs/model_card.md`) that includes:
- Model architecture (GraphSAGE with heterogeneous nodes)
- Training data description
- Performance metrics (AUC, AP)
- Known limitations
- Intended use cases

### 3.2 User Communication
- Users can view why a recipe was recommended
- Clear labeling: "Recommended for you" vs "Popular recipes"
- Option to see "Why was this recommended?"

### 3.3 Data Usage Transparency
- Privacy policy describes how interaction data is used
- Users can request their data export
- Clear opt-out mechanism for personalized recommendations

---

## 4. Privacy

### 4.1 Data Protection
- **Data Minimization**: Only collect necessary interaction data
- **Pseudonymization**: User IDs are hashed before storage
- **Data Retention**: 
  - Interaction logs retained for 12 months
  - Aggregated metrics retained indefinitely
  - Raw data purged after model training

### 4.2 Differential Privacy
```python
# Optional: Add differential privacy to training
def add_differential_privacy(gradients, epsilon=1.0, delta=1e-5):
    """Apply differential privacy noise to gradients."""
    sensitivity = compute_sensitivity(gradients)
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noisy_gradients = gradients + np.random.normal(0, noise_scale, gradients.shape)
    return noisy_gradients
```

### 4.3 Access Controls
- Object storage access requires authentication
- MLflow tracking server has role-based access
- Production systems use separate credentials from development

### 4.4 Implementation
- Credentials stored in Kubernetes Secrets / environment variables
- No PII in logs or model artifacts
- Data encrypted in transit (HTTPS) and at rest

---

## 5. Accountability

### 5.1 Audit Logging
All system actions are logged:
```python
AUDIT_LOG_SCHEMA = {
    'timestamp': 'ISO8601',
    'action': 'train|predict|rollback|deploy',
    'actor': 'user_id or system',
    'model_version': 'string',
    'details': 'dict',
    'outcome': 'success|failure',
}
```

### 5.2 Model Versioning
- All models tracked in MLflow Model Registry
- Version history maintained indefinitely
- Rollback capability to any previous version

### 5.3 Change Management
- All changes go through CI/CD pipeline
- Pull request reviews required
- Deployment approvals for production
- Rollback triggers documented

### 5.4 Incident Response
1. **Detection**: Automated monitoring alerts
2. **Response**: Auto-rollback for critical issues
3. **Investigation**: Audit logs preserved for analysis
4. **Remediation**: Root cause fix deployed through normal CI/CD
5. **Review**: Post-incident review documented

---

## 6. Robustness

### 6.1 Model Quality Gates
- Minimum AUC threshold: 0.70
- Minimum Average Precision: 0.65
- Models failing gates are not deployed

### 6.2 Production Monitoring
```yaml
# Prometheus alerting rules
groups:
  - name: ml-system-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: ModelLatencyHigh
        expr: histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          
      - alert: RecommendationQualityDegraded
        expr: rate(recommendations_clicked_total[1h]) / rate(recommendations_served_total[1h]) < 0.05
        for: 30m
        labels:
          severity: warning
```

### 6.3 Canary Deployments
- New models deployed to 10% traffic first
- Minimum 2-hour canary period
- Automatic rollback if error rate > 5%

### 6.4 Graceful Degradation
```python
def get_recommendations(user_id, model, fallback_enabled=True):
    """Get recommendations with fallback."""
    try:
        # Try ML-based recommendations
        recs = model.predict(user_id)
        return recs
    except ModelUnavailableError:
        if fallback_enabled:
            # Fallback to popularity-based recommendations
            return get_popular_recipes(limit=10)
        raise
```

### 6.5 Input Validation
- Validate all API inputs
- Sanitize user IDs and recipe IDs
- Rate limiting to prevent abuse

---

## 7. Monitoring Dashboard

Key metrics tracked in Grafana:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `model_prediction_latency_p99` | 99th percentile inference time | > 2 seconds |
| `recommendation_ctr` | Click-through rate on recommendations | < 5% |
| `model_version` | Current deployed model version | Change notification |
| `training_auc` | Model AUC from last training | < 0.70 |
| `error_rate_5xx` | Server error rate | > 5% |
| `data_drift_score` | Feature distribution drift | > 0.3 |

---

## 8. Responsible AI Checklist

Before each deployment, verify:

- [ ] Model passed all quality gates
- [ ] No significant drift in training data distribution
- [ ] Fairness metrics within acceptable range
- [ ] Explainability features functioning
- [ ] Privacy requirements met (no PII in logs)
- [ ] Audit logging enabled
- [ ] Rollback mechanism tested
- [ ] Monitoring alerts configured
- [ ] Documentation updated

---

## 9. Contact & Escalation

- **ML Team Lead**: [Contact for model issues]
- **Platform Team**: [Contact for infrastructure issues]
- **Security Team**: [Contact for security/privacy concerns]
- **Incident Hotline**: [For critical production issues]

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-20 | Team | Initial safeguarding plan |
