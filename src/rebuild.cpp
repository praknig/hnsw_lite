/**
 * @file rebuild.cpp
 * @brief Rebuilding an index while it stays in use; see HnswIndex::rebuild.
 */
#include <algorithm>
#include <optional>

#include <unordered_map>

#include "flat_index.h"
#include "hnsw_index.h"
#include "kmeans.h"

namespace vecdb {

RebuildStats HnswIndex::rebuild(const RebuildOptions& rebuild_options, const OperationOptions& options) {
    // 1. Under the lock: a snapshot of every record, and from now on every
    //    changed ID is recorded (note_change).
    struct Snapshot {
        std::uint64_t id;
        std::vector<float> vector;  // as given if kept, otherwise as stored (prepared)
        Metadata metadata;
        std::vector<std::uint8_t> code;  // codes only, codec kept: the code, copied as it is
    };
    std::vector<Snapshot> records;
    std::vector<float> sample;  // stored (prepared) vectors to train a codec on
    bool as_given = false, codes_only_now = false, retrain = false;
    std::unique_ptr<VectorCodec> kept_codec;
    Schema schema = Schema::dynamic();
    std::vector<std::string> fields;
    RebuildStats st;
    {
        const auto guard = lock_.write("rebuild");
        if (changed_) throw std::logic_error("rebuild: another rebuild of this index is running");
        const IdMap& ids = storage_.ids();
        as_given = storage_.originals() != nullptr;
        codes_only_now = codes_only();
        schema = metadata_.current_schema();
        fields = metadata_.indexed_fields();
        st.reclaimed = deleted_count();
        records.reserve(live_);
        // With codes only, the codec is never retrained (training needs the floats): each
        // code is carried over byte for byte rather than re-encoded from its decoded vector.
        const std::size_t cs = codes_only_now && codec_ && codec_->trained() && !as_given ? codec_->code_size() : 0;
        for (NodeId n = 0; n < storage_.size(); ++n)
            if (!ids.is_deleted(n))
                records.push_back({ids.external(n), as_given ? std::vector<float>(storage_.original(n).begin(), storage_.original(n).end()) : stored_copy(n),
                                   metadata_.read(n),
                                   cs ? std::vector<std::uint8_t>(codes_.begin() + std::ptrdiff_t(std::size_t(n) * cs), codes_.begin() + std::ptrdiff_t(std::size_t(n + 1) * cs))
                                      : std::vector<std::uint8_t>{}});
        if (codec_ && codec_->trained()) {
            retrain = rebuild_options.retrain && storage_.vectors().has_values();
            if (retrain) {
                const std::size_t d = storage_.vectors().dim(), take = std::min<std::size_t>(live_, 100000);
                sample.reserve(take * d);
                std::size_t i = 0;
                for (NodeId n = 0; n < storage_.size() && i < take; ++n)  // GCOVR_EXCL_BR_LINE: the sample is smaller only beyond 100,000 vectors
                    if (!ids.is_deleted(n) && (n % std::max<std::size_t>(1, live_ / take) == 0)) {  // GCOVR_EXCL_BR_LINE: likewise
                        const auto v = storage_.vectors().get(n);
                        sample.insert(sample.end(), v.begin(), v.end());
                        ++i;
                    }
            } else {
                kept_codec = codec_->clone();
            }
        }
        changed_.emplace();
    }
    auto stop_tracking = [&] {  // takes the lock: never call while holding it
        const auto guard = lock_.write("rebuild");
        changed_.reset();
    };

    // 2. Without any lock: build the new index from the snapshot. Everything that
    //    can fail is inside the try, so tracking always stops on a failure
    //    (found by rb07: constructing the new index could run out of memory).
    std::optional<HnswIndex> built;
    try {
        built.emplace(dim(), metric_, params_, schema);
        HnswIndex& fresh = *built;
        if (kept_codec) fresh.codec_ = std::move(kept_codec);
        else if (retrain) fresh.codec_->train(sample);
        if (codes_only_now) fresh.storage_.drop_vector_values();
        for (const std::string& f : fields) fresh.metadata_.index_field(f);
        detail::OperationControl control(options, records.size());
        for (const Snapshot& r : records) {
            fresh.code_given_ = r.code.empty() ? nullptr : r.code.data();
            fresh.add_unlocked(r.id, r.vector, r.metadata, !as_given);
            fresh.code_given_ = nullptr;
            control.advance();
        }
        control.finish();
    } catch (...) {
        stop_tracking();
        throw;
    }
    records.clear();
    records.shrink_to_fit();
    HnswIndex& fresh = *built;

    // 3. Under the lock: copy in the IDs changed meanwhile, then swap. Logged here,
    //    where it takes effect: after the changes made meanwhile (logged by them).
    const auto guard = lock_.write("rebuild");
    const ChangeLog::Scope logged(log_, log_.begin(walop::kRebuild, [&] { return walop::encode_rebuild(rebuild_options); }));
    try {
        const IdMap& ids = storage_.ids();
        for (std::uint64_t id : *changed_) {
            // fresh's own unlocked paths: this index's lock is held, and public calls
            // from inside an index operation are refused.
            if (const auto there = fresh.storage_.ids().find(id)) fresh.remove_slots({*there});
            if (const auto node = ids.find(id)) {
                if (storage_.originals()) fresh.add_unlocked(id, storage_.original(*node), metadata_.read(*node));
                else {  // codes only: the current code, carried over as it is
                    fresh.code_given_ = codes_only() && codec_ && codec_->trained() ? codes_.data() + std::size_t(*node) * codec_->code_size() : nullptr;
                    fresh.add_unlocked(id, stored_copy(*node), metadata_.read(*node), true);
                    fresh.code_given_ = nullptr;
                }
            }
            ++st.caught_up;
        }
        for (const std::string& f : metadata_.indexed_fields()) fresh.metadata_.index_field(f);  // indexed meanwhile
        // A catch-up that removed vectors is churn on the new graph (found by rb04:
        // reachability below 99% after busy writers): upkeep it before the swap.
        if (fresh.removals_since_upkeep_ + fresh.updates_since_upkeep_ > 0) (void)fresh.upkeep_unlocked({}, {});
    } catch (...) {
        changed_.reset();  // this index is untouched
        throw;
    }
    // Commit: moves only, nothing can throw.
    st.codec_retrained = retrain;
    codec_ = std::move(fresh.codec_);
    codes_ = std::move(fresh.codes_);
    storage_ = std::move(fresh.storage_);
    metadata_ = std::move(fresh.metadata_);
    rng_ = fresh.rng_;
    entry_ = fresh.entry_;
    max_level_ = fresh.max_level_;
    live_ = fresh.live_;
    free_slots_ = std::move(fresh.free_slots_);
    removals_since_upkeep_ = 0;  // a rebuilt graph carries no churn
    updates_since_upkeep_ = 0;
    visited_pool_.clear();
    changed_.reset();
    st.kept = live_;
    return st;
}

RebuildStats FlatIndex::rebuild(const RebuildOptions& rebuild_options, const OperationOptions& options) {
    // 1. Under the lock: the stored (prepared) vectors by ID; tracking starts.
    RebuildStats st;
    std::vector<std::uint64_t> snapshot_ids;
    std::vector<float> snapshot;  // rows of `d` values
    std::size_t d = 0, stride = 0, lists = 0, parts = 0;
    bool retrain_codec = false, retrain_lists = false;
    CodecKind kind = CodecKind::None;
    {
        const auto guard = lock_.write("rebuild");
        if (changed_) throw std::logic_error("rebuild: another rebuild of this index is running");
        d = vectors_.dim();
        stride = vectors_.stride();
        retrain_codec = rebuild_options.retrain && codec_ && codec_->trained() && vectors_.has_values();
        retrain_lists = rebuild_options.retrain && ivf_trained();
        if (live_ == 0) retrain_codec = retrain_lists = false;  // nothing to train on (k-means needs points)
        st.kept = live_;
        if (!retrain_codec && !retrain_lists) return st;  // rows stay packed: nothing else to rebuild
        if (retrain_codec) {
            kind = codec_->kind();
            if (const auto* pq = dynamic_cast<const PQCodec*>(codec_.get())) parts = pq->parts();
        }
        lists = lists_.size();
        snapshot_ids.reserve(live_);
        snapshot.reserve(std::size_t(live_) * d);
        for (NodeId row = 0; row < live_; ++row) {
            snapshot_ids.push_back(ids_.external(row));
            const std::vector<float> v = stored_copy(row);  // decoded if only codes are kept
            snapshot.insert(snapshot.end(), v.begin(), v.begin() + std::ptrdiff_t(d));
        }
        changed_.emplace();
    }
    auto stop_tracking = [&] {  // takes the lock: never call while holding it
        const auto guard = lock_.write("rebuild");
        changed_.reset();
    };

    // 2. Without any lock: train, then every snapshot vector's new code and list.
    const std::size_t n = snapshot_ids.size();
    std::unique_ptr<VectorCodec> codec;
    std::vector<std::uint8_t> codes;
    std::vector<float> centroids;     // lists x stride, padded
    std::vector<std::uint32_t> list_of;
    try {
        std::vector<float> sample;
        const std::size_t take = std::min<std::size_t>(n, 100000), step = std::max<std::size_t>(1, n / std::max<std::size_t>(take, 1));
        for (std::size_t i = 0; i < n && sample.size() < take * d; i += step)  // GCOVR_EXCL_BR_LINE: the sample is smaller only beyond 100,000 vectors
            sample.insert(sample.end(), snapshot.begin() + std::ptrdiff_t(i * d), snapshot.begin() + std::ptrdiff_t((i + 1) * d));
        if (retrain_codec) {
            codec = make_codec(kind, d, metric_, parts);
            codec->train(sample);
            codes.resize(n * codec->code_size());
        }
        if (retrain_lists) {
            const KMeansResult km = kmeans(sample, d, KMeansParams{std::min(lists, sample.size() / d), 25, 1, 0});
            const std::size_t k = km.centroids.size() / d;
            centroids.assign(k * stride, 0.0f);
            for (std::size_t c = 0; c < k; ++c)
                std::copy_n(km.centroids.begin() + std::ptrdiff_t(c * d), d, centroids.begin() + std::ptrdiff_t(c * stride));
            list_of.resize(n);
        }
        detail::OperationControl control(options, n);
        std::vector<float> padded(stride, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const std::span<const float> v(snapshot.data() + i * d, d);
            if (codec) codec->encode(v, codes.data() + i * codec->code_size());
            if (retrain_lists) {
                std::copy(v.begin(), v.end(), padded.begin());
                list_of[i] = nearest_of(centroids, padded.data());
            }
            control.advance();
        }
        control.finish();
    } catch (...) {
        stop_tracking();
        throw;
    }

    // 3. Under the lock: rows changed meanwhile anew, then everything at once.
    const auto guard = lock_.write("rebuild");
    const ChangeLog::Scope logged(log_, log_.begin(walop::kRebuild, [&] { return walop::encode_rebuild(rebuild_options); }));
    try {
        std::unordered_map<std::uint64_t, std::size_t> at;  // ID -> snapshot position
        at.reserve(n);
        for (std::size_t i = 0; i < n; ++i) at.emplace(snapshot_ids[i], i);
        const std::size_t cs = codec ? codec->code_size() : 0;
        std::vector<std::uint8_t> new_codes(codec ? std::size_t(live_) * cs : 0);
        std::vector<std::uint32_t> new_list_of(retrain_lists ? live_ : 0), new_pos(retrain_lists ? live_ : 0);
        std::vector<std::vector<NodeId>> new_lists(retrain_lists ? centroids.size() / stride : 0);
        std::vector<float> padded(stride, 0.0f);
        for (NodeId row = 0; row < live_; ++row) {
            const std::uint64_t id = ids_.external(row);
            const auto it = at.find(id);
            const bool fresh = it == at.end() || changed_->count(id);
            std::vector<float> now;
            if (fresh) {
                now = stored_copy(row);
                ++st.caught_up;
            }
            if (codec) {
                if (fresh) codec->encode(std::span<const float>(now.data(), d), new_codes.data() + std::size_t(row) * cs);
                else std::copy_n(codes.begin() + std::ptrdiff_t(it->second * cs), cs, new_codes.begin() + std::ptrdiff_t(std::size_t(row) * cs));
            }
            if (retrain_lists) {
                std::uint32_t list;
                if (fresh) {
                    std::copy_n(now.begin(), d, padded.begin());
                    list = nearest_of(centroids, padded.data());
                } else {
                    list = list_of[it->second];
                }
                new_list_of[row] = list;
                new_pos[row] = static_cast<std::uint32_t>(new_lists[list].size());
                new_lists[list].push_back(row);
            }
        }
        // Commit: moves only, nothing can throw.
        if (codec) {
            codec_ = std::move(codec);
            codes_ = std::move(new_codes);
            st.codec_retrained = true;
        }
        if (retrain_lists) {
            centroids_ = std::move(centroids);
            lists_ = std::move(new_lists);
            list_of_ = std::move(new_list_of);
            pos_ = std::move(new_pos);
            st.lists_retrained = true;
        }
    } catch (...) {
        changed_.reset();  // this index is untouched
        throw;
    }
    removals_since_upkeep_ = 0;
    updates_since_upkeep_ = 0;
    changed_.reset();
    st.kept = live_;
    return st;
}

}  // namespace vecdb

