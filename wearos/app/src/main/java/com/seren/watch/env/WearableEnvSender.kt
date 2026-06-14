package com.seren.watch.env

import android.content.Context
import android.util.Log
import com.google.android.gms.wearable.Wearable
import kotlinx.coroutines.tasks.await
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Ships ambient-light and GPS batches from the watch to the phone over the
 * Wearable Data Layer (MessageClient).
 *
 * Wire format (little-endian, version 1) — must match services/ai/envReceiver.ts:
 *   header : magic i32, version u16, reserved u16, count u32        (12 bytes)
 *   light  : count × (timestampMs i64 + lux f32)                    (12 bytes/sample)
 *   loc    : count × (timestampMs i64 + lat f64 + lon f64 + acc f32)(28 bytes/sample)
 *
 * Message paths:
 *   /seren/env/light     — ambient-light sample batches
 *   /seren/env/location  — GPS fix batches
 */
class WearableEnvSender(private val context: Context) {

    companion object {
        private const val TAG = "SerenEnv"
        // 4 ASCII chars read little-endian, matching the sleep sender's 'SRN1' convention.
        const val MAGIC_LIGHT: Int = 0x314C5253    // 'SRL1'
        const val MAGIC_LOCATION: Int = 0x31475253 // 'SRG1'
        const val VERSION: Short = 1
        const val PATH_LIGHT = "/seren/env/light"
        const val PATH_LOCATION = "/seren/env/location"
    }

    private val messageClient = Wearable.getMessageClient(context)
    private val nodeClient = Wearable.getNodeClient(context)

    suspend fun sendLight(samples: List<LightSample>) {
        if (samples.isEmpty()) return
        val headerSize = 4 + 2 + 2 + 4
        val buf = ByteBuffer.allocate(headerSize + samples.size * 12).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(MAGIC_LIGHT); buf.putShort(VERSION); buf.putShort(0); buf.putInt(samples.size)
        for (s in samples) { buf.putLong(s.timestampMs); buf.putFloat(s.lux) }
        deliver(PATH_LIGHT, buf.array(), samples.size)
    }

    suspend fun sendLocation(samples: List<LocationSample>) {
        if (samples.isEmpty()) return
        val headerSize = 4 + 2 + 2 + 4
        val buf = ByteBuffer.allocate(headerSize + samples.size * 28).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(MAGIC_LOCATION); buf.putShort(VERSION); buf.putShort(0); buf.putInt(samples.size)
        for (s in samples) {
            buf.putLong(s.timestampMs); buf.putDouble(s.latitude); buf.putDouble(s.longitude); buf.putFloat(s.accuracy)
        }
        deliver(PATH_LOCATION, buf.array(), samples.size)
    }

    private suspend fun deliver(path: String, payload: ByteArray, count: Int) {
        try {
            val nodes = nodeClient.connectedNodes.await()
            if (nodes.isEmpty()) {
                Log.w(TAG, "No paired nodes; $count samples on $path not delivered")
                return
            }
            for (node in nodes) messageClient.sendMessage(node.id, path, payload).await()
            Log.d(TAG, "Sent $count samples on $path to ${nodes.size} node(s)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send batch on $path", e)
        }
    }
}
