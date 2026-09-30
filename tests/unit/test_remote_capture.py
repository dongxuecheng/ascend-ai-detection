"""rtspGrpcServer 7abe4fd 协议与动态 SHM 回归，无需摄像头或昇腾设备。"""

import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stream import remote_capture as rc
from stream import stream_service_pb2 as pb


def write_shm(path, pixels, width, height, *, fmt=rc.PIXEL_BGR,
              flags=0, step=None, index=1, capacity=None, slot_count=3):
    payload = pixels.tobytes()
    slot_size = rc.align_up(128 + (capacity or len(payload)), 64)
    data = bytearray(slot_count * slot_size + 8)
    offset = (index % slot_count) * slot_size
    struct.pack_into("=Q", data, offset, 2)
    if step is None:
        step = width * (3 if fmt == rc.PIXEL_BGR else 2 if fmt == rc.PIXEL_YUYV422 else 1)
    struct.pack_into("=QQQQIIIII", data, offset + 64,
                     len(payload), width, height, 1234, 3 if fmt == rc.PIXEL_BGR else 1,
                     0, step, flags, fmt)
    data[offset + 128:offset + 128 + len(payload)] = payload
    struct.pack_into("=Q", data, slot_count * slot_size, index)
    path.write_bytes(data)
    return slot_size


class ShmTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        sem_patch = patch.object(rc, "_NotifySemaphore", side_effect=OSError("not available"))
        sem_patch.start()
        self.addCleanup(sem_patch.stop)
        self.pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    def reader(self, path, layout=None):
        reader = rc._ShmReader("test", layout)
        reader.shm_paths = [str(path)]
        self.addCleanup(reader.close)
        self.assertTrue(reader.connect())
        return reader

    def test_dynamic_size_overrides_default_layout_and_copies_pixels(self):
        path = self.directory / "stream"
        size = write_shm(path, self.pixels, 4, 2, flags=1)
        reader = self.reader(path, rc._compute_shm_layout(6_220_800))
        self.assertEqual(reader.SLOT_SIZE, size)
        ok, frame, ts = reader.read()
        self.assertTrue(ok)
        self.assertEqual(ts, 1234)
        self.assertTrue(reader.last_frame_corrupted)
        self.assertEqual(reader.last_frame_meta["pix_fmt"], rc.PIXEL_BGR)
        np.testing.assert_array_equal(frame, self.pixels)
        self.assertFalse(reader.read()[0])
        reader.close()
        np.testing.assert_array_equal(frame, self.pixels)
        self.assertIsNone(reader.last_frame_meta)

    def test_negotiated_slot_count(self):
        path = self.directory / "stream"
        write_shm(path, self.pixels, 4, 2, slot_count=4)
        layout = rc._compute_shm_layout(1024)
        layout["slot_count"] = 4
        reader = self.reader(path, layout)
        np.testing.assert_array_equal(reader.read()[1], self.pixels)

    def test_padded_bgr_rows(self):
        path = self.directory / "stream"
        padded = np.zeros((2, 16), np.uint8)
        padded[:, :12] = self.pixels.reshape(2, 12)
        write_shm(path, padded, 4, 2, step=16)
        np.testing.assert_array_equal(self.reader(path).read()[1], self.pixels)

    def test_invalid_file_closes_fd_and_does_not_map(self):
        path = self.directory / "bad"
        path.write_bytes(b"invalid")
        reader = rc._ShmReader("test")
        self.addCleanup(reader.close)
        reader.shm_paths = [str(path)]
        with patch.object(rc.os, "close", wraps=os.close) as close:
            self.assertFalse(reader.connect())
        close.assert_called_once()
        self.assertIsNone(reader._mmap_obj)

    def test_layout_rejects_invalid_sizes(self):
        for size in (0, 8, 128, 3 * 128 + 8, 3 * 193 + 8):
            with self.subTest(size=size):
                self.assertIsNone(rc.derive_shm_layout_from_size(size))

    def test_inode_replacement_and_resize_reset_sequence(self):
        for capacity in (64, 512):
            with self.subTest(capacity=capacity):
                old, new = self.directory / "old", self.directory / "new"
                write_shm(old, self.pixels, 4, 2, capacity=64, index=5)
                write_shm(new, self.pixels + 1, 4, 2, capacity=capacity, index=0)
                reader = self.reader(old)
                self.assertTrue(reader.read()[0])
                # Windows 不允许替换已映射文件，用新路径模拟同名 SHM 的新 inode。
                reader.shm_paths = [str(new)]
                reader._last_progress = time.monotonic() - 1
                reader._last_probe = 0
                ok, frame, _ = reader.read(blocking=True, timeout_ms=100)
                self.assertTrue(ok)
                np.testing.assert_array_equal(frame, self.pixels + 1)
                self.assertEqual(reader._last_idx, 0)
                reader.close()

    def test_sequence_lock_checked_after_copy(self):
        path = self.directory / "stream"
        write_shm(path, self.pixels, 4, 2)
        reader = self.reader(path)
        copied = False
        rebuild, read_u64 = reader._rebuild_frame, reader._read_u64

        def copy_frame(*args):
            nonlocal copied
            image = rebuild(*args)
            copied = True
            return image

        def sequence(offset):
            value = read_u64(offset)
            return value + 2 if copied and offset != reader.HEAD_IDX_OFFSET else value

        with patch.object(reader, "_rebuild_frame", side_effect=copy_frame), patch.object(
            reader, "_read_u64", side_effect=sequence
        ):
            self.assertIsNone(reader.retrieve()[0])
        self.assertEqual(reader._last_idx, -1)

    def test_negative_timeout_means_wait_and_semaphore_retry(self):
        reader = rc._ShmReader("test")
        self.addCleanup(reader.close)
        with patch.object(reader, "_try_read", side_effect=[(False, None, 0), (True, self.pixels, 1)]), patch.object(
            rc.time, "sleep"
        ):
            self.assertTrue(reader.read(blocking=True, timeout_ms=-1)[0])
        semaphore = Mock()
        reader._next_sem_attach = 0
        with patch.object(rc, "_NotifySemaphore", return_value=semaphore):
            self.assertTrue(reader._try_attach_notify_sem())

    def test_yuv_formats_and_bgr_conversion(self):
        for fmt, shape, code in (
            (rc.PIXEL_NV12, (3, 4), cv2.COLOR_YUV2BGR_NV12),
            (rc.PIXEL_I420, (3, 4), cv2.COLOR_YUV2BGR_I420),
            (rc.PIXEL_YUYV422, (2, 8), cv2.COLOR_YUV2BGR_YUY2),
        ):
            with self.subTest(fmt=fmt):
                path = self.directory / str(fmt)
                pixels = np.full(shape, 128, np.uint8)
                pixels.flat[0] = 90
                write_shm(path, pixels, 4, 2, fmt=fmt)
                reader = self.reader(path)
                frame = reader.read()[1]
                np.testing.assert_array_equal(frame, pixels)
                packed = pixels.reshape(2, 4, 2) if fmt == rc.PIXEL_YUYV422 else pixels
                np.testing.assert_array_equal(rc.frame_to_bgr(frame, fmt), cv2.cvtColor(packed, code))

    def test_invalid_frame_metadata(self):
        reader = rc._ShmReader("test")
        self.addCleanup(reader.close)
        meta = dict(w=4, h=2, ch=3, depth=0, step=12, pix_fmt=rc.PIXEL_BGR)
        for update in ({"w": 0}, {"step": 1}, {"depth": 7}, {"ch": 0},
                       {"pix_fmt": 99}, {"pix_fmt": rc.PIXEL_NV12, "w": 3},
                       {"pix_fmt": rc.PIXEL_NV12, "h": 3}):
            with self.subTest(update=update):
                self.assertIsNone(reader._rebuild_frame(self.pixels.tobytes(), meta | update))


class GrpcTests(unittest.TestCase):
    def setUp(self):
        self.client = rc.RTSPClient("example:50051")
        self.client._stub = Mock()
        self.addCleanup(self.client.disconnect)
        self.stub = self.client._stub
        self.stub.StartStream.return_value = pb.StartResponse(success=True, stream_id="s")
        self.stub.CheckStream.return_value = pb.CheckResponse(
            stream=pb.StreamInfo(stream_id="s", status=rc.STATUS_CONNECTED)
        )

    def test_protocol_field_numbers(self):
        self.assertEqual(pb.StartRequest.DESCRIPTOR.fields_by_name["pixel_format"].number, 9)
        self.assertEqual(pb.FrameResponse.DESCRIPTOR.fields_by_name["corrupted"].number, 5)
        for number, name in enumerate(("fps", "media_lag_ms", "corrupted_frames", "glitch_ratio", "pixel_format"), 12):
            self.assertEqual(pb.StreamInfo.DESCRIPTOR.fields_by_name[name].number, number)
        self.assertEqual(pb.StartRequest().pixel_format, rc.PIXEL_BGR)
        self.assertFalse(pb.FrameResponse().corrupted)

    def test_start_and_restart_preserve_pixel_format(self):
        self.assertEqual(self.client.start_stream("rtsp://example/video", use_shared_mem=True,
                                                pixel_format=rc.PIXEL_NV12), "s")
        request = self.stub.StartStream.call_args.args[0]
        self.assertEqual(request.pixel_format, rc.PIXEL_NV12)
        self.stub.CheckStream.return_value.stream.status = rc.STATUS_NOT_FOUND
        self.stub.StartStream.return_value.stream_id = "new"
        self.assertEqual(self.client._get_current_stream_id("s"), "new")
        self.assertEqual(self.stub.StartStream.call_args.args[0], request)
        self.stub.StartStream.return_value.stream_id = "newer"
        self.assertEqual(self.client._get_current_stream_id("s"), "newer")

    def test_health_fields(self):
        info = pb.StreamInfo(stream_id="s", fps=24.5, media_lag_ms=150,
                             corrupted_frames=3, glitch_ratio=0.25, pixel_format=rc.PIXEL_NV12)
        self.stub.CheckStream.return_value = pb.CheckResponse(stream=info)
        self.stub.ListStreams.return_value = pb.ListStreamsResponse(streams=[info])
        for result in (self.client.check_stream("s"), self.client.list_streams()[0]):
            for field in ("fps", "media_lag_ms", "corrupted_frames", "glitch_ratio", "pixel_format"):
                self.assertEqual(result[field], getattr(info, field))

    def test_jpeg_corrupted_and_legacy_pair(self):
        pixels = np.full((2, 4, 3), 128, np.uint8)
        data = cv2.imencode(".jpg", pixels)[1].tobytes()
        self.stub.GetLatestFrame.return_value = pb.FrameResponse(
            success=True, image_data=data, frame_seq=5, corrupted=True
        )
        with patch.object(rc, "_HAS_TURBOJPEG", False):
            ts, frame, corrupted = self.client.read_ex("s")
            self.assertEqual(ts, 5)
            self.assertTrue(corrupted)
            np.testing.assert_array_equal(frame, pixels)
            self.assertEqual(len(self.client.read("s")), 2)
            self.stub.StreamFrames.return_value = [self.stub.GetLatestFrame.return_value]
            self.assertTrue(next(self.client.stream_frames_ex("s"))[2])
            self.assertEqual(len(next(self.client.stream_frames("s"))), 2)

    def test_shm_metadata_uses_current_stream_id(self):
        self.client._stream_id_map["s"] = "new"
        reader = Mock(last_frame_meta={"pix_fmt": rc.PIXEL_I420}, last_frame_corrupted=True)
        reader.read.return_value = (True, np.zeros((3, 4), np.uint8), 1234)
        self.client._shm_readers["new"] = reader
        self.client._stream_modes["new"] = True
        self.assertTrue(self.client.read_ex("s")[2])
        meta = self.client.get_last_frame_meta("s")
        self.assertEqual(meta["pix_fmt"], rc.PIXEL_I420)
        meta["pix_fmt"] = 99
        self.assertEqual(reader.last_frame_meta["pix_fmt"], rc.PIXEL_I420)

    def test_jpeg_fallback_for_pyturbojpeg_and_decode_error(self):
        pixels = np.full((2, 4, 3), 128, np.uint8)
        data = cv2.imencode(".jpg", pixels)[1].tobytes()
        for backend in (object(), Mock(decompress=Mock(side_effect=RuntimeError("decode failed")))):
            with self.subTest(backend=type(backend)), patch.object(rc, "_HAS_TURBOJPEG", True), patch.object(
                rc, "turbojpeg", backend
            ):
                np.testing.assert_array_equal(self.client._decode_jpeg(data), pixels)


if __name__ == "__main__":
    unittest.main()
