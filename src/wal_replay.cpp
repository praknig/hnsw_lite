/**
 * @file wal_replay.cpp
 * @brief Attaching a write-ahead log to an index, and replaying one (durability, D2).
 */
#include <unordered_set>

#include "flat_index.h"
#include "hnsw_index.h"
#include "serialize.h"
#include "wal_ops.h"

namespace vecdb {

namespace {

/// Replays records onto an index: the operations both index types have, plus
/// `own(op)` for the others (false: a type this index cannot replay).
template <class Index, class Own>
bool replay_records(Index& index, std::span<const WalRecord> records, Own&& own) {
    std::unordered_set<std::uint64_t> aborted;
    for (const WalRecord& r : records)
        if (r.type == walop::kAbort) aborted.insert(walop::decode(r.type, r.payload).lsn);
    // The last operation not aborted: the only one a stop can have left unfinished.
    std::uint64_t final_lsn = 0;
    for (const WalRecord& r : records)
        if (r.type != walop::kAbort && !aborted.count(r.lsn)) final_lsn = r.lsn;
    for (const WalRecord& r : records) {
        if (r.type == walop::kAbort || aborted.count(r.lsn)) continue;
        const walop::Op op = walop::decode(r.type, r.payload);
        // Logged and not aborted: it succeeded when it was made, so it must again.
        try {
            switch (op.type) {
                case walop::kAdd: index.add(op.ids.at(0), op.vectors, op.metadata.at(0)); break;
                case walop::kAddBatch:
                    if (op.metadata.empty()) index.add_batch(op.ids, op.vectors, 1);
                    else index.add_batch(op.ids, op.vectors, op.metadata, 1);
                    break;
                case walop::kRemove: (void)index.remove(op.ids.at(0)); break;
                case walop::kRemoveBatch: (void)index.remove_batch(op.ids); break;
                case walop::kUpdate:
                    if (op.metadata.empty()) (void)index.update(op.ids.at(0), op.vectors);
                    else (void)index.update(op.ids.at(0), op.vectors, op.metadata[0]);
                    break;
                case walop::kSetMetadata:
                case walop::kSetMetadataIds:
                    for (std::uint64_t id : op.ids) (void)index.set_metadata(id, op.metadata.at(0));
                    break;
                case walop::kRebuild: (void)index.rebuild(op.rebuild); break;
                case walop::kTrainCodec: index.train_codec(); break;
                case walop::kCreatePayloadIndex: index.create_payload_index(op.field); break;
                default:
                    if (!own(op)) throw FormatError("log record of a type this index cannot replay");
            }
        } catch (const FormatError&) {  // GCOVR_EXCL_START: no index operation throws FormatError
            throw;
        }  // GCOVR_EXCL_STOP
        catch (const std::exception& e) {
            // The final operation may have been in progress when the process stopped:
            // logged, failed, and stopped before its abort record. Anywhere else, the
            // log does not match the index.
            if (r.lsn == final_lsn) return true;
            throw FormatError("replaying log record " + std::to_string(r.lsn) + " failed: " + e.what() +
                              " (the log does not match the index it is replayed onto)");
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- HnswIndex

void HnswIndex::attach_log(WalWriter log) {
    const auto guard = lock_.write("attach_log");
    if (durable_) throw std::logic_error("attach_log: a durable index keeps its own log");
    log_.attach(std::move(log));
}

std::optional<WalWriter> HnswIndex::detach_log() {
    const auto guard = lock_.write("detach_log");
    if (durable_) throw std::logic_error("detach_log: a durable index keeps its own log");
    return log_.detach();
}

bool HnswIndex::replay(std::span<const WalRecord> records) {
    {
        const auto guard = lock_.read();
        if (log_.attached()) throw std::logic_error("replay: detach the log first (replayed changes must not be logged again)");
    }
    return replay_records(*this, records, [&](const walop::Op& op) {
        switch (op.type) {
            case walop::kCompact: (void)compact(); return true;
            case walop::kReorder: (void)reorder(); return true;
            case walop::kUpkeep: (void)upkeep(op.upkeep); return true;
            case walop::kSetPlanner: set_planner_params(op.planner); return true;
            case walop::kSetAutoCompact: set_auto_compact(op.fraction); return true;
            default: return false;
        }
    });
}

// ---------------------------------------------------------------- FlatIndex

void FlatIndex::attach_log(WalWriter log) {
    const auto guard = lock_.write("attach_log");
    if (durable_) throw std::logic_error("attach_log: a durable index keeps its own log");
    log_.attach(std::move(log));
}

std::optional<WalWriter> FlatIndex::detach_log() {
    const auto guard = lock_.write("detach_log");
    if (durable_) throw std::logic_error("detach_log: a durable index keeps its own log");
    return log_.detach();
}

bool FlatIndex::replay(std::span<const WalRecord> records) {
    {
        const auto guard = lock_.read();
        if (log_.attached()) throw std::logic_error("replay: detach the log first (replayed changes must not be logged again)");
    }
    return replay_records(*this, records, [&](const walop::Op& op) {
        if (op.type != walop::kTrainLists) return false;
        train_lists();
        return true;
    });
}

}  // namespace vecdb