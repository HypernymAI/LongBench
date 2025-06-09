# Gemini API Tier Upgrade Process for Production Benchmarking

**Date**: June 8, 2025  
**Project ID**: gen-lang-client-0191283286  
**Author**: Production Engineering Team  
**Purpose**: Business validation and customer onboarding benchmarks

## Executive Summary

This document outlines the process for upgrading Google Gemini API quotas from paid_tier to paid_tier_3 to enable production-scale benchmarking for customer validation. Current quotas are insufficient for running industry-standard benchmarks required for enterprise sales demonstrations.

## Current State Analysis

### Project Configuration
- **Project ID**: gen-lang-client-0191283286
- **API Endpoint**: https://generativelanguage.googleapis.com/v1beta/
- **Current Tier**: paid_tier (basic paid quotas)

### Current Quota Limits

| Model | Current Tier | Tokens/Min | Requests/Min | Required Tier | Target Tokens/Min |
|-------|--------------|------------|--------------|---------------|-------------------|
| gemini-2.0-flash | paid_tier | 4,000,000 | 2,000 | paid_tier_3 | 30,000,000 |
| gemini-2.5-flash | paid_tier | 1,000,000 | 1,000 | paid_tier_3 | 8,000,000 |
| gemini-2.5-pro | paid_tier | 2,000,000 | 150 | paid_tier_3 | 8,000,000 |
| gemini-1.5-flash | paid_tier | 4,000,000 | 2,000 | paid_tier | 4,000,000 |
| gemini-1.5-pro | paid_tier | 4,000,000 | 1,000 | paid_tier | 4,000,000 |

### Quota Metric Details
- **Primary Metric**: `generativelanguage.googleapis.com/generate_content_paid_tier_input_token_count`
- **Target Metric**: `generativelanguage.googleapis.com/generate_content_paid_tier_3_input_token_count`
- **Unit**: `1/min/{project}/{model}`

## Business Justification

### Use Case: Customer Validation Benchmarks
- **Benchmark**: LongBench-v2 (industry-standard long-context evaluation)
- **Scale**: 503 examples, average 10,000+ tokens per example
- **Processing**: Requires parallel execution to meet customer demonstration timelines
- **Impact**: Direct revenue impact - blocking customer onboarding and sales closure

### Current Limitations Impact
1. **Rate Limiting**: Constant 429 errors with current 4M token/minute limit
2. **Time Impact**: Benchmarks take 10x longer than acceptable for customer demos
3. **Business Impact**: Unable to demonstrate performance at scale required by enterprise customers

## Upgrade Process Steps

### Step 1: Verify Current Quotas
```bash
# Check current quotas for all Gemini models
gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/gen-lang-client-0191283286 \
  | grep -E "(gemini|generate_content_paid_tier)" > current_quotas.txt
```

### Step 2: Document Required Changes

**Quota Upgrade Request Details**:
```
Service: generativelanguage.googleapis.com
Project: gen-lang-client-0191283286

Requested Changes:
1. Upgrade gemini-2.0-flash from paid_tier to paid_tier_3
   - Current: 4M tokens/min → Target: 30M tokens/min
   
2. Upgrade gemini-2.5-flash from paid_tier to paid_tier_3
   - Current: 1M tokens/min → Target: 8M tokens/min
   
3. Upgrade gemini-2.5-pro from paid_tier to paid_tier_3
   - Current: 2M tokens/min → Target: 8M tokens/min

Business Justification:
- Customer validation benchmarks for enterprise sales
- Industry-standard performance testing (LongBench-v2)
- Immediate need for scheduled customer demonstrations
- Sufficient GCP credits allocated for this purpose
```

### Step 3: Submit Quota Increase Request

Since `gcloud support` commands are not available, use the Google Cloud Console:

1. **Navigate to Quotas Page**:
   ```
   https://console.cloud.google.com/iam-admin/quotas?project=gen-lang-client-0191283286
   ```

2. **Filter for Gemini Quotas**:
   - Search: `generativelanguage`
   - Look for: `generate_content_paid_tier_input_token_count`

3. **Select Models to Upgrade**:
   - ☑ gemini-2.0-flash
   - ☑ gemini-2.5-flash
   - ☑ gemini-2.5-pro

4. **Click "EDIT QUOTAS"** and submit with business justification

### Step 4: Alternative - Direct Support Ticket

If quota edit is not available, create support ticket via Console:

1. Go to: https://console.cloud.google.com/support/cases
2. Create new case with:
   - **Type**: Technical issue
   - **Category**: APIs & Services
   - **Component**: Gemini API
   - **Priority**: P2 (Business Impact)

**Ticket Template**:
```
Subject: Urgent: Gemini API Tier Upgrade Required for Customer Validation

Project ID: gen-lang-client-0191283286

We need immediate upgrade from paid_tier to paid_tier_3 for Gemini API models to support critical customer validation benchmarks.

Business Impact:
- Blocking enterprise customer demonstrations scheduled this week
- Unable to run industry-standard benchmarks (LongBench-v2) at required scale
- Direct revenue impact - preventing sales closure

Technical Requirements:
- Upgrade gemini-2.0-flash: 4M → 30M tokens/min
- Upgrade gemini-2.5-flash: 1M → 8M tokens/min  
- Upgrade gemini-2.5-pro: 2M → 8M tokens/min

Current paid_tier quotas cause constant rate limiting (429 errors) making customer demonstrations impossible.

We have allocated GCP credits specifically for this validation work.

Please expedite as we have customer commitments this week.
```

### Step 5: Verify Upgrade

After approval (typically 24-48 hours):

```bash
# Verify new quotas
gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/gen-lang-client-0191283286 \
  | grep -A5 "paid_tier_3" | grep -B5 "gemini-2.0-flash"

# Test with production workload
python pred.py -m gemini-2.0-flash -n 16 --seed 42
```

## Implementation Details

### Configuration Updates Required

No code changes needed. The API automatically uses the highest available tier quotas.

### Benchmark Execution Plan

Once upgraded to tier 3:

```bash
# Production benchmark run
cd /Users/fieldempress/Desktop/source/hypernym/LongBench
source venv/bin/activate

# Run benchmarks for customer validation
python pred.py -m gemini-2.0-flash -n 16 --seed 42     # Primary model
python pred.py -m gemini-2.5-flash -n 8 --seed 42      # Comparison model
python pred.py -m gemini-2.5-pro -n 4 --seed 42        # Premium model

# Generate customer report
python result.py > customer_benchmark_results.txt
```

### Expected Performance

With tier 3 quotas:
- **Processing Speed**: 16x improvement (4 processes → 16 processes)
- **Completion Time**: ~30 minutes vs 8+ hours
- **Customer Impact**: Same-day results vs multi-day wait

## Risk Mitigation

### Quota Monitoring
```bash
# Monitor usage during benchmarks
watch -n 60 'gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/gen-lang-client-0191283286 \
  | grep -A2 "gemini-2.0-flash" | grep "usage"'
```

### Fallback Plan
If tier 3 upgrade is delayed:
1. Run with 4 processes instead of 16
2. Use caching to avoid re-running completed examples
3. Prioritize specific benchmark subsets for customer demos

## Cost Considerations

### Estimated Usage
- **Benchmark Size**: ~5.7M tokens per complete run
- **Cost at Gemini 2.0 Flash rates**: ~$0.58 per run
- **Customer Validation Runs**: ~10 runs = $5.80 total
- **Well within allocated GCP credits**

### Cost Optimization
- Caching implemented to avoid duplicate API calls
- Results stored for reuse across customer demonstrations
- Deterministic seeds ensure reproducible results

## Success Criteria

1. **Technical**: Complete LongBench-v2 in under 1 hour
2. **Business**: Enable same-day benchmark results for customers
3. **Performance**: Achieve >99% completion rate without rate limiting

## Timeline

- **Day 0**: Submit quota increase request
- **Day 1-2**: Google approval process
- **Day 2-3**: Verify upgrade and run validation tests
- **Day 3+**: Production customer benchmarks

## Appendix: Technical Commands

### Check Specific Quota
```bash
gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/gen-lang-client-0191283286 \
  --filter="metric.value:generate_content_paid_tier*" \
  --format="table(quota.metric,quota.limit,quota.usage)"
```

### Monitor Active Requests
```bash
# Watch for rate limit errors
tail -f results/gemini-*.jsonl | grep -E "(429|Error)"
```

### Generate Benchmark Report
```bash
# After completion
python result.py
python documentation/generate_customer_report.py
```

---

**Document Version**: 1.0  
**Last Updated**: June 8, 2025  
**Next Review**: After tier upgrade approval