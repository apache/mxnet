# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import unittest
from unittest.mock import MagicMock, mock_open, patch

from artifact_repository import *


class TestArtifactRepositoryTool(unittest.TestCase):

    @staticmethod
    def create_argparse_namespace(libmxnet_path: Optional[str] = 'path_to_libmxnet',
                                  git_sha: Optional[str] = 'abc123',
                                  variant: Optional[str] = 'cpu',
                                  operating_system: Optional[str] = 'linux',
                                  libtype: Optional[str] = 'static',
                                  bucket: Optional[str] = 's3bucket',
                                  licenses: Optional[List[str]] = [],
                                  dependencies: Optional[List[str]] = []) -> argparse.Namespace:
        """
        Returns a namespace object containing the script's arguments with sample or specified values
        """
        ns = argparse.Namespace()
        ns.libmxnet = libmxnet_path
        ns.git_sha = git_sha
        ns.variant = variant
        ns.os = operating_system
        ns.libtype = libtype
        ns.bucket = bucket
        ns.licenses = licenses
        ns.dependencies = dependencies

        return ns

    @patch('artifact_repository.check_output')
    def test_get_commit_id_from_cmd_returns_none_on_fail(self, mock):
        """
        Tests get_commit_id_from_cmd returns None if the command fails
        """
        not_sucessful = 255
        mock.side_effect = CalledProcessError(cmd="some command", returncode=not_sucessful)
        self.assertIsNone(get_commit_id_from_cmd())

    def test_probe_commit_id_mxnet_sha(self):
        """
        Tests the value of MXNET_SHA env var is returned
        """
        with patch.dict('os.environ', {'MXNET_SHA': 'abcd1234'}):
            self.assertEqual(probe_commit_id(), 'abcd1234')

    def test_probe_commit_id_git_commit(self):
        """
        Tests the value of GIT_COMMIT env. var is returned
        if MXNET_SHA env var is not present
        """
        with patch.dict('os.environ', {'GIT_COMMIT': 'abcd1234'}):
            self.assertEqual(probe_commit_id(), 'abcd1234')

        with patch.dict('os.environ', {'MXNET_SHA': 'efgh5678', 'GIT_COMMIT': 'abcd1234'}):
            self.assertEqual(probe_commit_id(), 'efgh5678')

    @patch('artifact_repository.get_commit_id_from_cmd')
    def test_probe_commit_id_git_cmd(self, mock):
        """
        Tests the git commit id from the git command is returned
        if neither MXNET_SHA nor GIT_COMMIT env vars are set
        """
        mock.return_value = 'abcd1234'
        self.assertEqual(probe_commit_id(), 'abcd1234')

    def test_get_linux_os_release_properties(self):
        properties = """
        KEY=value
        KEY2=value2
        KEY3=value3
        """
        mock = mock_open(read_data=properties)
        with(patch('artifact_repository.os.path')) as path_mock:
            path_mock.is_file.return_value = True
            with patch('artifact_repository.open', mock, create=True):
                properties = get_linux_os_release_properties()
        self.assertEqual(properties['KEY3'], 'value3')

    def test_get_linux_os_release_properties_with_quotes(self):
        """
        Tests quote marks are removed from values
        """
        properties = """
        KEY="value"
        """
        mock = mock_open(read_data=properties)
        with(patch('artifact_repository.os.path')) as path_mock:
            path_mock.is_file.return_value = True
            with patch('artifact_repository.open', mock, create=True):
                properties = get_linux_os_release_properties()
        self.assertEqual(properties['KEY'], 'value')

    @patch('artifact_repository.sys')
    def test_probe_operating_system_windows(self, mock):
        mock.platform = 'win32'
        self.assertEqual(probe_operating_system(), 'win32')

    @patch('artifact_repository.sys')
    def test_probe_operating_system_darwin(self, mock):
        mock.platform = 'darwin'
        self.assertEqual(probe_operating_system(), 'darwin')

    @patch('artifact_repository.sys')
    @patch('artifact_repository.get_linux_os_release_properties')
    def test_probe_operating_system_linux(self, mock_props, mock_sys):
        mock_props.return_value = {'ID': 'ubuntu', 'VERSION_ID': '16.04'}

        mock_sys.platform = 'linux'
        self.assertEqual(probe_operating_system(), 'ubuntu16.04')

        # sys.platform can return linux or linux2
        mock_sys.platform = 'linux2'
        self.assertEqual(probe_operating_system(), 'ubuntu16.04')

    @patch('artifact_repository.check_output')
    def test_get_cuda_version(self, mock):
        """
        Tests correct cuda version with the right format is returned
        :return:
        """
        mock.return_value = b'Cuda compilation tools, release 10.0, V10.0.130'
        cuda_version = get_cuda_version()
        self.assertEqual(cuda_version, '100')

        mock.return_value = b'Cuda compilation tools, release 9.2, V9.2.148'
        cuda_version = get_cuda_version()
        self.assertEqual(cuda_version, '92')

    @patch('artifact_repository.check_output')
    def test_get_cuda_version_not_found(self, mock):
        """
        Tests None is returned there's an error retrieving the cuda version
        :return:
        """
        not_sucessful = 255
        mock.side_effect = CalledProcessError(cmd="nvidia version command", returncode=not_sucessful)
        self.assertIsNone(get_cuda_version())

    @patch('artifact_repository.get_libmxnet_features')
    def test_probe_variant_native(self, mock_features):
        """
        Tests 'native' is returned if MKLDNN and CUDA features are OFF
        """
        mock_features.return_value = {'MKLDNN': False, 'CUDA': False}
        self.assertEqual(probe_mxnet_variant('libmxnet.so'), 'native')

    @patch('artifact_repository.get_libmxnet_features')
    def test_probe_variant_cpu(self, mock_features):
        """
        Tests 'cpu' is returned if MKLDNN is ON and CUDA is OFF
        """
        mock_features.return_value = {'MKLDNN': True, 'CUDA': False}
        self.assertEqual(probe_mxnet_variant('libmxnet.so'), 'cpu')

    @patch('artifact_repository.get_libmxnet_features')
    @patch('artifact_repository.get_cuda_version')
    def test_probe_variant_cuda(self, mock_cuda_version, mock_features):
        """
        Tests 'cu100' is returned if MKLDNN is OFF and CUDA is ON and CUDA version is 10.0
        """
        mock_features.return_value = {'MKLDNN': True, 'CUDA': True}
        mock_cuda_version.return_value = '100'
        self.assertEqual(probe_mxnet_variant('libmxnet.so'), 'cu100')

    @patch('artifact_repository.get_libmxnet_features')
    def test_probe_variant_cuda_returns_none_on_no_features(self, mock_features):
        """
        Tests None is returned if the mxnet features could not be extracted from the libmxnet.so file
        """
        mock_features.return_value = None
        self.assertIsNone(probe_mxnet_variant('libmxnet.so'))

    @patch('artifact_repository.get_libmxnet_features')
    @patch('artifact_repository.get_cuda_version')
    def test_probe_variant_cuda_mkl(self, mock_cuda_version, mock_features):
        """
        Tests exception is raised if CUDA feature is ON but cuda version could not be determined
        """
        mock_features.return_value = {'MKLDNN': True, 'CUDA': True}
        mock_cuda_version.return_value = None
        with self.assertRaises(RuntimeError):
            probe_mxnet_variant('libmxnet.so')

    def test_probe_artifact_repository_bucket(self):
        """
        Tests artiact repository bucket is retrieved from environment variable ARTIFACT_REPOSITORY_BUCKET
        """
        with patch.dict('os.environ', {'ARTIFACT_REPOSITORY_BUCKET': 'some bucket'}):
            self.assertEqual(probe_artifact_repository_bucket(), 'some bucket')

    @patch('artifact_repository.probe_commit_id')
    def test_probe_no_commit_id(self, mock):
        """
        Tests commit id gets probed if not set by user
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(git_sha=None)
        mock.return_value = 'deadbeef'
        probe(fake_args)
        mock.assert_called_once()
        self.assertEqual(fake_args.git_sha, 'deadbeef')

    @patch('artifact_repository.probe_commit_id')
    def test_probe_no_commit_id_failed(self, mock):
        """
        Tests script will exit if commid id probe fails
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(git_sha=None)
        mock.return_value = None
        with self.assertRaises(SystemExit):
            probe(fake_args)


    @patch('artifact_repository.probe_operating_system')
    def test_probe_no_operating_system(self, mock):
        """
        Tests operating system gets probed if not set by user
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(operating_system=None)
        mock.return_value = 'be/os'
        probe(fake_args)
        mock.assert_called_once()
        self.assertEqual(fake_args.os, 'be/os')

    @patch('artifact_repository.probe_operating_system')
    def test_probe_no_operating_system_failed(self, mock):
        """
        Tests script will exit if operating system probe fails
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(operating_system=None)
        mock.return_value = None
        with self.assertRaises(SystemExit):
            probe(fake_args)

    @patch('artifact_repository.probe_mxnet_variant')
    def test_probe_no_variant(self, mock):
        """
        Tests mxnet variant gets probed if not set by user
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(variant=None)
        mock.return_value = 'cpu90mkl'
        probe(fake_args)
        mock.assert_called_once()
        self.assertEqual(fake_args.variant, 'cpu90mkl')

    @patch('artifact_repository.probe_mxnet_variant')
    def test_probe_no_mxnet_variant_failed(self, mock):
        """
        Tests script will exit if mxnet variant probe fails
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(variant=None)
        mock.return_value = None
        with self.assertRaises(SystemExit):
            probe(fake_args)

    @patch('artifact_repository.probe_artifact_repository_bucket')
    def test_probe_no_bucket(self, mock):
        """
        Tests artifact repository bucket gets probed if not set by user
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(bucket=None)
        mock.return_value = 'bucket'
        probe(fake_args)
        mock.assert_called_once()
        self.assertEqual(fake_args.bucket, 'bucket')

    @patch('artifact_repository.probe_artifact_repository_bucket')
    def test_probe_no_bucket_failed(self, mock):
        """
        Tests script will exit if bucket probe fails
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(bucket=None)
        mock.return_value = None
        with self.assertRaises(SystemExit):
            probe(fake_args)

    def test_get_s3_key_prefix(self):
        """
        Tests S3 key prefix is properly formated
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(git_sha="abc123",
                                                                         operating_system='linux',
                                                                         variant='cpu',
                                                                         libtype='static')

        self.assertEqual(get_s3_key_prefix(fake_args), 'abc123/static/linux/cpu/')

    def test_get_s3_key_prefix_with_subdir(self):
        """
        Tests S3 key prefix with sub-directory is properly formated
        """
        fake_args = TestArtifactRepositoryTool.create_argparse_namespace(git_sha="abc123",
                                                                         operating_system='linux',
                                                                         variant='cpu',
                                                                         libtype='static')

        self.assertEqual(get_s3_key_prefix(fake_args, subdir='subdir'), 'abc123/static/linux/cpu/subdir/')

    @patch('artifact_repository.s3')
    def test_try_s3_download_fails_on_bad_response(self, mock_s3):
        """
        Tests RuntimeError is thrown if the response is malformed
        """
        key_prefix = 'some/key/prefix'
        mock_s3.list_objects_v2.return_value = {
            'something': 'not quite right'
        }

        with self.assertRaises(RuntimeError):
            try_s3_download(bucket='bucket', s3_key_prefix=key_prefix, destination='')

    @patch('artifact_repository.s3')
    def test_try_s3_download_returns_false_on_no_keys(self, mock_s3):
        """
        Tests False is returned when there are no keys for the prefix. Ie. no artifact to download
        """
        key_prefix = 'some/key/prefix'
        mock_s3.list_objects_v2.return_value = {
            'KeyCount': 0
        }
        self.assertFalse(try_s3_download(bucket='bucket', s3_key_prefix=key_prefix, destination=''))

    @patch('artifact_repository.os.makedirs', autospec=True)
    @patch('artifact_repository.s3')
    def test_try_s3_download_with_destination(self, mock_s3, mock_makedirs):
        """
        Tests files are downloaded to the right destinations when destination parameter is not empty
        """
        key_prefix = 'some/key/prefix'
        s3_keys = [
            {'Key': '{}/file.txt'.format(key_prefix)},
            {'Key': '{}/subdir/other.txt'.format(key_prefix)},
            {'Key': '{}/another/sub/dir/f.txt'.format(key_prefix)}
        ]

        mock_s3.list_objects_v2.return_value = {
            'Contents': s3_keys,
            'KeyCount': 3
        }

        mock_s3.download_fileobj = MagicMock()
        mock_fopen = mock_open()

        with patch('artifact_repository.open', mock_fopen, create=True):
            dest = os.path.join('dest', 'ination')
            self.assertTrue(try_s3_download(bucket='bucket', s3_key_prefix=key_prefix, destination=dest))

            # Assert directories are created
            mock_makedirs.assert_has_calls([
                unittest.mock.call(dest, exist_ok=True),
                unittest.mock.call(os.path.join(dest, 'subdir'), exist_ok=True),
                unittest.mock.call(os.path.join(dest, 'another', 'sub', 'dir'), exist_ok=True)
            ])

            # Assert files are downloaded
            mock_s3.download_fileobj.assert_has_calls([
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[0]['Key']),
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[1]['Key']),
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[2]['Key']),
            ])

    @patch('artifact_repository.os.makedirs', autospec=True)
    @patch('artifact_repository.s3')
    def test_try_s3_download(self, mock_s3, mock_makedirs):
        """
        Tests files are downloaded to the right destinations when destination parameter is empty
        """
        key_prefix = 'some/key/prefix'
        s3_keys = [
            {'Key': '{}/file.txt'.format(key_prefix)},
            {'Key': '{}/subdir/other.txt'.format(key_prefix)},
            {'Key': '{}/another/sub/dir/f.txt'.format(key_prefix)}
        ]

        mock_s3.list_objects_v2.return_value = {
            'Contents': s3_keys,
            'KeyCount': 3
        }

        mock_s3.download_fileobj = MagicMock()
        mock_fopen = mock_open()

        with patch('artifact_repository.open', mock_fopen, create=True):
            dest = ''
            self.assertTrue(try_s3_download(bucket='bucket', s3_key_prefix=key_prefix, destination=dest))

            # Assert directories are created
            mock_makedirs.assert_has_calls([
                unittest.mock.call(dest, exist_ok=True),
                unittest.mock.call(os.path.join(dest, 'subdir'), exist_ok=True),
                unittest.mock.call(os.path.join(dest, 'another', 'sub', 'dir'), exist_ok=True)
            ])

            # Assert files are downloaded
            mock_s3.download_fileobj.assert_has_calls([
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[0]['Key']),
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[1]['Key']),
                unittest.mock.call(Bucket='bucket', Fileobj=unittest.mock.ANY, Key=s3_keys[2]['Key']),
            ])

    @patch('artifact_repository.s3')
    def test_s3_upload(self, mock_s3):
        """
        Tests files are uploaded using the supplied s3_key_prefix
        """
        key_prefix = 'some/key/prefix'
        paths = [
            os.path.join('mainfile.txt'),
            os.path.join('some/dir/file.txt'),
            os.path.join('some/other/dir/another.txt'),
        ]

        mock_s3.upload_fileobj = MagicMock()
        mock_fopen = mock_open(read_data=b'some data')

        with patch('artifact_repository.open', mock_fopen, create=True):
            s3_upload(bucket='bucket', s3_key_prefix=key_prefix, paths=paths)
            mock_s3.upload_fileobj.assert_has_calls([
                unittest.mock.call(Fileobj=unittest.mock.ANY, Key='some/key/prefix/mainfile.txt', Bucket='bucket'),
                unittest.mock.call(Fileobj=unittest.mock.ANY, Key='some/key/prefix/file.txt', Bucket='bucket'),
                unittest.mock.call(Fileobj=unittest.mock.ANY, Key='some/key/prefix/another.txt', Bucket='bucket'),
            ])

    @patch('artifact_repository.os.path.isfile')
    @patch('artifact_repository.os.path.exists')
    def test_is_file_is_file(self, mock_exists, mock_isfile):
        """
        Tests is file returns True when path exists and is a file
        """
        mock_exists.return_value = True
        mock_isfile.return_value = True
        self.assertTrue(is_file('some/path'))

    @patch('artifact_repository.os.path.isfile')
    @patch('artifact_repository.os.path.exists')
    def test_is_file_not_file(self, mock_exists, mock_isfile):
        """
        Tests is file returns False when path exists and is _not_ a file
        """
        mock_exists.return_value = True
        mock_isfile.return_value = False
        self.assertFalse(is_file('some/path'))

    @patch('artifact_repository.os.path.exists')
    def test_is_file_not_found(self, mock_exists):
        """
        Tests FileNotFound error thrown if file not found
        """
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError) as ctx:
            is_file('some/path')
        self.assertEqual(str(ctx.exception), 'File \'{}\' not found'.format('some/path'))

    def test_sanitize_path_array_empty_paths(self):
        """
        Tests empty paths are removed
        """
        self.assertListEqual(sanitize_path_array([' ', '\t', '     \n']), [])

    @patch('artifact_repository.is_file')
    @patch('artifact_repository.glob', autospec=True)
    def test_sanitize_path_array_directories(self, mock_glob, mock_isfile):
        """
        Tests directory paths are removed
        """
        mock_isfile.side_effect = [False, True, False]
        mock_glob.glob = lambda x: [x]
        self.assertListEqual(sanitize_path_array(['dir1', 'file', 'dir2']), ['file'])

    def test_write_libmxnet_meta(self):
        """
        Tests libmxnet.meta is properly written out
        """
        mock_fopen = mock_open()
        with patch('artifact_repository.open', mock_fopen, create=True):
            fake_args = TestArtifactRepositoryTool.create_argparse_namespace(git_sha='abcd1234',
                                                                             variant='gpu',
                                                                             operating_system='lunix',
                                                                             libtype='stynamic')
            write_libmxnet_meta(args=fake_args, destination='dest')
            mock_fopen.assert_called_once_with(os.path.join('dest', 'libmxnet.meta'), 'w')
            mock_fopen().write.called_with(
                'commit_id: abcd1234\ndependency_linking: stynamic\nos: lunix\nvariant: gpu\n')

    def test_push_artifact_throws_no_license_error(self):
        """
        Tests push artifact throwns error if no licenses are defined
        """
        args = TestArtifactRepositoryTool.create_argparse_namespace(licenses=[])
        with self.assertRaises(RuntimeError) as ctx:
            push_artifact(args)
        self.assertEqual(str(ctx.exception),
                          "No licenses defined. Please submit the licenses to be shipped with the binary.")

        args = args = TestArtifactRepositoryTool.create_argparse_namespace(licenses=None)
        with self.assertRaises(RuntimeError) as ctx:
            push_artifact(args)
        self.assertEqual(str(ctx.exception),
                          "No licenses defined. Please submit the licenses to be shipped with the binary.")
