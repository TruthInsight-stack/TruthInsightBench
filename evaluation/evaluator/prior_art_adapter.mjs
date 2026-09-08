import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function normalizedId(value) {
  return String(value ?? '').trim().toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, '');
}

function hitKeys(hit) {
  return new Set([
    normalizedId(hit?.id),
    normalizedId(hit?.doi),
    ...(hit?.alternate_ids ?? []).map(normalizedId),
  ].filter(Boolean));
}

function reviewedCandidateWasRetrieved(candidate, queries) {
  const wanted = new Set([
    normalizedId(candidate?.provider_id),
    normalizedId(candidate?.doi),
  ].filter(Boolean));
  if (wanted.size === 0) return false;
  return queries.some((query) => (query?.hits ?? []).some((hit) => {
    const keys = hitKeys(hit);
    return [...wanted].some((key) => keys.has(key));
  }));
}

export function hitStrictlyBefore(hit, t0) {
  const date = String(hit?.publication_date ?? hit?.date ?? '').trim();
  if (/^\d{4}(?:-\d{2}(?:-\d{2})?)?$/.test(date)) return date < t0;
  const year = Number(hit?.publication_year ?? hit?.year);
  return Number.isInteger(year) && year < Number(t0.slice(0, 4));
}

export function normalizeFrozenNoveltyAsset({ taskId, t0, goldCardsPath, assetPath }) {
  const assetBytes = fs.readFileSync(assetPath);
  const goldBytes = fs.readFileSync(goldCardsPath);
  const asset = JSON.parse(assetBytes);
  const gold = JSON.parse(goldBytes);
  if (gold.schema !== 'truthinsightbench-validation-anchors') throw new Error(`${taskId}: prior-art adapter only accepts the published validation anchors`);
  if (asset.schema !== 'truthinsightbench-frozen-novelty-search-evidence') throw new Error(`${taskId}: prior-art asset is not the published schema`);
  const querySpecPath = path.join(path.dirname(assetPath), 'novelty_query_spec.json');
  const querySpecBytes = fs.readFileSync(querySpecPath);
  const querySpec = JSON.parse(querySpecBytes);
  const querySpecHash = sha256(querySpecBytes);
  const goldCards = Array.isArray(gold.cards) ? gold.cards : [];
  const assetCards = asset.cards && typeof asset.cards === 'object' ? asset.cards : {};
  const cardIds = goldCards.map((card) => card.card_id);

  const cards = cardIds.map((cardId) => {
    const goldCard = goldCards.find((card) => card.card_id === cardId);
    const assetCard = assetCards[cardId];
    const executedQueries = assetCard?.queries ?? [];
    const expectedQueries = querySpec.cards?.[cardId]?.queries ?? [];
    const executedQueryText = executedQueries.map((query) => query.query);
    const expectedQueryText = expectedQueries.map(String);
    const queryText = new Set(executedQueryText);
    const hits = executedQueries.flatMap((query) => query.hits ?? []);
    const scoredHits = hits.filter((hit) => hitStrictlyBefore(hit, t0));

    return {
      card_id: cardId,
      expected_query_count: expectedQueries.length,
      executed_query_count: executedQueries.length,
      all_expected_queries_present: expectedQueryText.every((query) => queryText.has(query)),
      executed_queries_match_spec: executedQueryText.length === expectedQueryText.length
        && expectedQueryText.every((query) => queryText.has(query)),
      all_queries_succeeded: executedQueries.length > 0 && executedQueries.every((query) => query.ok === true || query.request_completed === true),
      all_query_cutoffs_match_t0: executedQueries.every((query) => query.strict_before === t0),
      returned_hit_count: hits.length,
      scored_pre_t0_hit_count: scoredHits.length,
      excluded_non_pre_t0_hit_count: hits.length - scoredHits.length,
      all_scored_hits_pre_t0: scoredHits.every((hit) => hitStrictlyBefore(hit, t0)),
      reviewed_candidate_count: 0,
      all_reviewed_candidates_in_snapshot: true,
      curator_verdict: null,
    };
  });

  const cardSetMatches = Object.keys(assetCards).sort().join('\0') === [...cardIds].sort().join('\0');
  const referenceCardReady = asset.strict_before === t0
    && asset.query_spec_sha256 === querySpecHash
    && cardSetMatches
    && cards.every((card) => card.executed_queries_match_spec
      && card.all_queries_succeeded
      && card.all_query_cutoffs_match_t0
      && card.all_scored_hits_pre_t0
      && card.all_reviewed_candidates_in_snapshot);

  return {
    schema: 'truthinsightbench-frozen-prior-art-adapter',
    task_id: taskId,
    t0,
    source_schema: asset.schema,
    asset_path: assetPath,
    asset_sha256: sha256(assetBytes),
    query_spec_path: querySpecPath,
    query_spec_sha256: querySpecHash,
    query_spec_hash_matches_asset: asset.query_spec_sha256 === querySpecHash,
    gold_cards_path: goldCardsPath,
    gold_cards_sha256: sha256(goldBytes),
    strict_before_matches_t0: asset.strict_before === t0,
    card_set_matches: cardSetMatches,
    cards,
    reference_card_prior_art_ready: referenceCardReady,
    claim_specific_prior_art_ready: false,
    target_exclusion_verified: false,
    scoring_contract: {
      allowed_use: '可用于核验发现与冻结参考卡所代表中央主张的截止日前先例重合。',
      forbidden_inference: '不能仅凭两张参考卡的查询覆盖，宣称任意外围发现均不存在先例。',
      missing_claim_specific_search: '对参考卡之外的主张，评价器须生成主张级查询并冻结结果；缺失时该动作记 judge_failed，而不是给参赛者 0 分。',
      target_exclusion: '只有另有目标论文标识符拒绝清单并完成命中核验时，才可把 target_exclusion_verified 设为 true。',
    },
  };
}
