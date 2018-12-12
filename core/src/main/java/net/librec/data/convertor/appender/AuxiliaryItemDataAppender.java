package net.librec.data.convertor.appender;

/**
 * @author szkb
 * @date 2018/12/11 16:27
 */
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import net.librec.conf.Configuration;
import net.librec.conf.Configured;
import net.librec.data.DataAppender;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import org.apache.commons.lang.StringUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class AuxiliaryItemDataAppender extends Configured implements DataAppender {

    /**
     * The size of the buffer
     */
    private static final int BSIZE = 1024 * 1024;

    /**
     * a {@code SparseMatrix} object build by the social data
     */
    private SequentialAccessSparseMatrix userSocialMatrix;

    /**
     * The path of the appender data file
     */
    private String inputDataPath;

    /**
     * User {raw id, inner id} map from rating data
     */
    private BiMap<String, Integer> userIds;

    /**
     * Item {raw id, inner id} map from rating data
     */
    private BiMap<String, Integer> itemIds;

    // todo 注释 这里还要初始化
    private HashMap<Integer, ArrayList<Integer>> itemFeature = new HashMap<>();


    /**
     * Initializes a newly created {@code SocialDataAppender} object with null.
     */
    public AuxiliaryItemDataAppender() {
        this(null);
    }

    /**
     * Initializes a newly created {@code SocialDataAppender} object with a
     * {@code Configuration} object
     *
     * @param conf {@code Configuration} object for construction
     */
    public AuxiliaryItemDataAppender(Configuration conf) {
        this.conf = conf;
    }

    /**
     * Process appender data.
     *
     * @throws IOException if I/O error occurs during processing
     */
    @Override
    public void processData() throws IOException {
        if (conf != null && StringUtils.isNotBlank(conf.get("data.appender.path"))) {
            inputDataPath = conf.get("dfs.data.dir") + "/" + conf.get("data.appender.path");
            readData(inputDataPath);
        }
    }

    /**
     * Read data from the data file. Note that we didn't take care of the
     * duplicated lines.
     *
     * @param inputDataPath the path of the data file
     * @throws IOException if I/O error occurs during reading
     */
    private void readData(String inputDataPath) throws IOException {
        // Table {row-id, col-id, rate}
        Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
        // BiMap {raw id, inner id} userIds, itemIds
        final List<File> files = new ArrayList<File>();
        final ArrayList<Long> fileSizeList = new ArrayList<Long>();
        SimpleFileVisitor<Path> finder = new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                fileSizeList.add(file.toFile().length());
                files.add(file.toFile());
                return super.visitFile(file, attrs);
            }
        };
        Files.walkFileTree(Paths.get(inputDataPath), finder);
        long allFileSize = 0;
        for (Long everyFileSize : fileSizeList) {
            allFileSize = allFileSize + everyFileSize.longValue();
        }
        // loop every dataFile collecting from walkFileTree
        for (File dataFile : files) {
            FileInputStream fis = new FileInputStream(dataFile);
            FileChannel fileRead = fis.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
            int len;
            String bufferLine = new String();
            byte[] bytes = new byte[BSIZE];
            while ((len = fileRead.read(buffer)) != -1) {
                buffer.flip();
                buffer.get(bytes, 0, len);
                bufferLine = bufferLine.concat(new String(bytes, 0, len)).replaceAll("\r", "\n");
                String[] bufferData = bufferLine.split("(\n)+");
                boolean isComplete = bufferLine.endsWith("\n");
                int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
                for (int i = 0; i < loopLength; i++) {
                    String line = new String(bufferData[i]);
                    ArrayList<Integer> feature = new ArrayList<>();
                    String[] data = line.trim().split("\\|\\|");
                    String[] str1 = data[0].trim().split("\\|");
                    String itemId = str1[0];
                    if (itemIds.containsKey(itemId)) {
                        int item = itemIds.get(itemId);
                        // todo 这里注意有两个斜杠
                        String[] str2 = data[1].trim().split("\\|");
                        for (int j = 1; j < str2.length; j++) {
                            feature.add(Integer.valueOf(str2[j]));
                        }
                        itemFeature.put(item, feature);
                    }

                }
                if (!isComplete) {
                    bufferLine = bufferData[bufferData.length - 1];
                }
                buffer.clear();
            }
            fileRead.close();
            fis.close();
        }
        int numRows = userIds.size(), numCols = userIds.size();
        // build rating matrix
        userSocialMatrix = new SequentialAccessSparseMatrix(numRows, numCols, dataTable);
        // release memory of data table
        dataTable = null;
    }

    public HashMap<Integer, ArrayList<Integer>> getItemFeature() {
        return itemFeature;
    }
    /**
     * Get user appender.
     *
     * @return the {@code SparseMatrix} object built by the social data.
     */
    public SequentialAccessSparseMatrix getUserAppender() {
        return userSocialMatrix;
    }

    /**
     * Get item appender.
     *
     * @return null
     */
    public SequentialAccessSparseMatrix getItemAppender() {
        return null;
    }

    /**
     * Set user mapping data.
     *
     * @param userMappingData user {raw id, inner id} map
     */
    @Override
    public void setUserMappingData(BiMap<String, Integer> userMappingData) {
        this.userIds = userMappingData;
    }

    /**
     * Set item mapping data.
     *
     * @param itemMappingData item {raw id, inner id} map
     */
    @Override
    public void setItemMappingData(BiMap<String, Integer> itemMappingData) {
        this.itemIds = itemMappingData;
    }
}
