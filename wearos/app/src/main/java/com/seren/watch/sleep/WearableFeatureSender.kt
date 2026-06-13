package com.seren.watch.sleep

import android.content.Context
import android.util.Log
import com.google.android.gms.wearable.Wearable
import kotlinx.coroutines.tasks.await
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Ships completed sleep epochs from the watch to the phone over the Wearable
 * Data Layer (MessageClient).
 *
 * Wire format (little-endian, version 1):
 *   header  : magic 'SRN1' (4 bytes), version u16, reserved u16
 *   metadata: captureStartMs i64, epochCount u32, featuresPerEpoch u16, reserved u16
 *   payload : epochCount × (startMs i64 + 11 × float32)
 *
 * Total bytes per epoch = 8 + 11*4 = 52.  A 9-hour night ≈ 1080 epochs ≈ 56 KB.
 *
 * Message path:
 *   /seren/sleep/features/batch     — incremental batches during the night
 *   /seren/sleep/features/finalize  — last batch + "session ended" signal
 */
class WearableFeatureSender(private val context: Context) {

    companion object {
        private const val TAG = "SerenSleep"
        const val MAGIC: Int = 0x314E5253 // 'SRN1' little-endian
        const val VERSION: Short = 1
        const val PATH_BATCH = "/seren/sleep/features/batch"
        const val PATH_FINALIZE = "/seren/sleep/features/finalize"
    }

    private val messageClient = Wearable.getMessageClient(context)
    private val nodeClient = Wearable.getNodeClient(context)

    /** Sends `epochs` to every connected node. Use [final] = true at session end. */
    suspend fun send(captureStartMs: Long, epochs: List<RawEpochFeatures>, final: Boolean) {
        if (epochs.isEmpty() && !final) return
        val payload = encode(captureStartMs, epochs)
        val path = if (final) PATH_FINALIZE else PATH_BATCH
        try {
            val nodes = nodeClient.connectedNodes.await()
            if (nodes.isEmpty()) {
                Log.w(TAG, "No paired nodes; ${epochs.size} epochs not delivered")
                return
            }
            for (node in nodes) {
                messageClient.sendMessage(node.id, path, payload).await()
            }
            Log.d(TAG, "Sent ${epochs.size} epochs (final=$final) to ${nodes.size} node(s)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send epoch batch", e)
        }
    }

    private fun encode(captureStartMs: Long, epochs: List<RawEpochFeatures>): ByteArray {
        val headerSize = 4 + 2 + 2 + 8 + 4 + 2 + 2
        val epochSize = 8 + RawEpochFeatures.SIZE * 4
        val buf = ByteBuffer.allocate(headerSize + epochs.size * epochSize)
            .order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(MAGIC)
        buf.putShort(VERSION)
        buf.putShort(0)
        buf.putLong(captureStartMs)
        buf.putInt(epochs.size)
        buf.putShort(RawEpochFeatures.SIZE.toShort())
        buf.putShort(0)
        for (e in epochs) {
            buf.putLong(e.startMs)
            for (v in e.features) buf.putFloat(v)
        }
        return buf.array()
    }
}
