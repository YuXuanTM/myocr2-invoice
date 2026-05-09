import re

AMOUNT_RE = re.compile(r"^[¥￥]?\s*\d+(?:\.\d{1,2})?$")
DATE_RE = re.compile(r"\d{4}年\d{2}月\d{2}日")
TIME_RE = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})")
TIME_VALUE_RE = re.compile(r"^(?:[01]?\d|2[0-3])[:：][0-5]\d(?:\s*开)?$")


def _clean_text(v):
    if not isinstance(v, str):
        return v
    return v.strip()


def postprocess_invoice_result(result):
    if not isinstance(result, dict):
        return result

    for k, v in list(result.items()):
        result[k] = _clean_text(v)

    title = result.get('title', '')
    if isinstance(title, str):
        if '铁路电子客票' in title and '（' not in title and '(' not in title:
            # 修复常见漏识别左括号
            result['title'] = title.replace('铁路电子客票', '（铁路电子客票')

    # 时间字段规范化: 15: 06开 -> 15:06开
    if 'time_of_departure' in result and isinstance(result.get('time_of_departure'), str):
        dep_time = result.get('time_of_departure')
        cleaned_time = TIME_RE.sub(r'\1:\2', dep_time).strip()
        if cleaned_time and TIME_VALUE_RE.match(cleaned_time):
            result['time_of_departure'] = cleaned_time
        else:
            result.pop('time_of_departure', None)

    # 铁路票经常开票日期和乘车日期同日，issue_date 漏识别时做兜底
    if not result.get('issue_date') and '铁路电子客票' in result.get('title', ''):
        date_of_departure = result.get('date_of_departure', '')
        if isinstance(date_of_departure, str) and DATE_RE.search(date_of_departure):
            result['issue_date'] = DATE_RE.search(date_of_departure).group(0)

    # 非金额文本误识别到金额字段时清空，避免脏数据
    tax_exclusive = result.get('tax_exclusive_total_amount', '')
    if isinstance(tax_exclusive, str) and tax_exclusive and not AMOUNT_RE.match(tax_exclusive):
        if any(flag in tax_exclusive for flag in ['改签', '始发', '终到', '退票']):
            result['tax_exclusive_total_amount'] = ''

    return result
