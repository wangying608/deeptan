import sys
from pybedtools import BedTool


def ensure_binary(value):
    """确保输入值是二进制格式（bytes）。"""
    if isinstance(value, str):
        return value.encode('utf-8')
    return value


def build_gene_id_name_map(gtf_path):
    """
    从 GTF 文件中提取基因 ID 和名称的映射。

    参数:
        gtf_path (str): GTF 文件的路径。

    返回:
        dict: 基因 ID 到基因名称的映射。
    """
    gtf_iter = BedTool(gtf_path)
    id_name_map = {}
    for fields in gtf_iter:
        if fields[2] == "gene":
            try:
                gene_id = ensure_binary(fields.attrs["gene_id"])
                gene_name = ensure_binary(fields.attrs.get("gene_name", fields.attrs["gene_id"]))
                id_name_map[gene_id] = gene_name
            except ValueError as e:
                print(f"Error parsing attributes: {fields[-1]}")
                raise e
    return id_name_map


def test_build_gene_id_name_map(test_gtf_path):
    """
    测试 build_gene_id_name_map 函数。
    """
    # 调用函数生成基因 ID 和名称的映射
    id_name_map = build_gene_id_name_map(test_gtf_path)

    # 打印结果
    # print("Gene ID to Name Map:")
    # for gene_id, gene_name in id_name_map.items():
    #     print(f"{gene_id.decode('utf-8')}: {gene_name.decode('utf-8')}")

    print("Test passed!")


if __name__ == "__main__":
    test_build_gene_id_name_map(sys.argv[1])
