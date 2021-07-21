package org.apache.mxnet.repository;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.mxnet.util.FilenameUtils;
import org.apache.mxnet.util.Utils;
import org.apache.mxnet.util.ZipUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;

public class Repository {

    private static final Logger logger = LoggerFactory.getLogger(Repository.class);

    private String name;
    private URI uri;
    private Path resourceDir;

    Repository(String name, String uri) {
        setName(name);
        setUri(URI.create(uri));
    }

    Repository(Item item) {
        this(item.getName(), item.getUrl());
    }

    public static Path initRepository(Item item) throws IOException {
        Repository repository = new Repository(item);
        repository.prepare();
        return repository.getLocalDir();
    }

    private void setResourceDir(Path resourceDir) {
        this.resourceDir = resourceDir;
    }

    private Path getResourceDir() {
        return resourceDir;
    }

    public Path getLocalDir() {
        return getResourceDir().resolve(getName());
    }

    public void setUri(URI uri) {
        this.uri = uri;
    }

    public URI getUri() {
        return uri;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void prepare() throws IOException {
        // TODO: specify the resource path according to Artifact's properties, including name, version and so on.
        String uriPath = getUri().getPath();
        if (uriPath != null && !"".equals(uriPath) && uriPath.charAt(0) == '/') {
            uriPath = uriPath.substring(1);
        }
        Path resourceDir = getCacheDirectory().resolve(uriPath);
        setResourceDir(resourceDir);
        if (Files.exists(getResourceDir())) {
            logger.debug("Files have been downloaded already: {}", getResourceDir());
            return;
        }
        Path parentDir = getResourceDir().toAbsolutePath().getParent();
        if (parentDir == null) {
            throw new AssertionError(String.format("Parent path should never be null: {}", getResourceDir().toString()));
        }

        Files.createDirectories(parentDir);
        Path tmp = Files.createTempDirectory(parentDir, getResourceDir().toFile().getName());

        //dismiss Progress related

        try {
            logger.debug("Repository to download: {}", getUri().toString());
            download(tmp);
            Utils.moveQuietly(tmp, getResourceDir());
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }

    private void download(Path tmp) throws IOException {
        String name = getName();
        URI fileUri = getUri();
        logger.debug("Downloading artifact: {} at {}...", name, fileUri);
        try (InputStream is = fileUri.toURL().openStream()) {
            String extension = FilenameUtils.getFileType(fileUri.getPath());
            save(is, tmp, name, extension, isArchiveFile(extension));
        }
    }

    private boolean isArchiveFile(String fileType) {
        return "tgz".equals(fileType) || "zip".equals(fileType) || "tar".equals(fileType);
    }

    protected void save(
            InputStream is, Path tmp, String repoName, String extension, boolean archive)
            throws IOException {
//        ProgressInputStream pis = new ProgressInputStream(is);

        if (archive) {
            Path diretory;
            if (!repoName.isEmpty()) {
                // honer the name set in metadata.json
                diretory = tmp.resolve(repoName);
                Files.createDirectories(diretory);
            } else {
                diretory = tmp;
            }
            if ("zip".equals(extension)) {
                ZipUtils.unzip(is, diretory);
            } else if ("tgz".equals(extension)) {
                untar(is, diretory, true);
            } else if ("tar".equals(extension)) {
                untar(is, diretory, false);
            } else {
                throw new IOException("File type is not supported: " + extension);
            }
        } else {
            Path file = tmp.resolve(repoName);
            if ("zip".equals(extension)) {
                ZipInputStream zis = new ZipInputStream(is);
                zis.getNextEntry();
                Files.copy(zis, file, StandardCopyOption.REPLACE_EXISTING);
            } else if ("gzip".equals(extension)) {
                Files.copy(new GZIPInputStream(is), file, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
            }
        }
//        pis.validateChecksum(item);
    }


    private void untar(InputStream is, Path dir, boolean gzip) throws IOException {
        InputStream bis;
        if (gzip) {
            bis = new GzipCompressorInputStream(new BufferedInputStream(is));
        } else {
            bis = new BufferedInputStream(is);
        }
        try (TarArchiveInputStream tis = new TarArchiveInputStream(bis)) {
            TarArchiveEntry entry;
            while ((entry = tis.getNextTarEntry()) != null) {
                String entryName = entry.getName();
                if (entryName.contains("..")) {
                    throw new IOException("Malicious zip entry: " + entryName);
                }
                Path file = dir.resolve(entryName).toAbsolutePath();
                if (entry.isDirectory()) {
                    Files.createDirectories(file);
                } else {
                    Path parentFile = file.getParent();
                    if (parentFile == null) {
                        throw new AssertionError(
                                "Parent path should never be null: " + file.toString());
                    }
                    Files.createDirectories(parentFile);
                    Files.copy(tis, file, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    public Path getCacheDirectory() throws IOException {
        Path dir = Utils.getCacheDir().resolve("cache/repo");
        if (Files.notExists(dir)) {
            Files.createDirectories(dir);
        } else if (!Files.isDirectory(dir)) {
            throw new IOException("Failed initialize cache directory: " + dir.toString());
        }
        return dir;
    }


}
