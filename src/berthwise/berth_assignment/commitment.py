from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from math import ceil
from typing import Self

from ..dwell_estimator.contracts import CompletedCall
from ..dwell_estimator.neighbors import learn, predict_dwell

DEFAULT_PORT_BUFFER = 60


@dataclass(frozen=True, slots=True)
class IssuedCommitment:
    call: str
    line: str
    berth: str
    moored: int
    promised: int
    issued: int

    def __post_init__(self):
        if not isinstance(self.call, str) or not self.call.strip():
            raise ValueError("承诺航次缺失")
        if not isinstance(self.line, str) or not self.line.strip():
            raise ValueError("承诺航线缺失")
        if not isinstance(self.berth, str) or not self.berth.strip():
            raise ValueError("承诺泊位缺失")
        if type(self.moored) is not int or not 0 <= self.moored <= 30000:
            raise ValueError("承诺靠泊分钟越界")
        if type(self.promised) is not int or not 0 <= self.promised <= 30000:
            raise ValueError("承诺离泊分钟越界")
        if self.moored >= self.promised:
            raise ValueError("承诺离泊须晚于靠泊")
        if type(self.issued) is not int or not 0 <= self.issued <= 30000:
            raise ValueError("承诺签发分钟越界")

    @classmethod
    def parse(cls, row: dict) -> Self:
        try:
            call = row["call"]
            call = call.strip() if isinstance(call, str) else call
            line = row["line"]
            line = line.strip() if isinstance(line, str) else line
            berth = row["berth"]
            berth = berth.strip() if isinstance(berth, str) else berth
            moored = row["moored"]
            promised = row["promised"]
            issued = row["issued"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"缺少可用的业务字段 {error}") from error
        return cls(call=call, line=line, berth=berth, moored=moored,
                   promised=promised, issued=issued)

    def as_record(self) -> dict:
        return asdict(self)


def _as_commitment(raw) -> IssuedCommitment:
    return raw if isinstance(raw, IssuedCommitment) else IssuedCommitment.parse(raw)


def _merge(marks):
    ordered = []
    for begin, end in sorted(marks):
        if begin >= end:
            raise ValueError("承诺占泊时钟必须递增")
        if ordered and ordered[-1][1] >= begin:
            ordered[-1] = (ordered[-1][0], max(ordered[-1][1], end))
        else:
            ordered.append((begin, end))
    return tuple(ordered)


def locked_blocks(issued, moment):
    if type(moment) is not int or not 0 <= moment <= 30000:
        raise ValueError("开班分钟越界")
    seen = set()
    blocks = defaultdict(list)
    for raw in issued:
        row = _as_commitment(raw)
        if row.call in seen:
            raise ValueError("承诺航次重复")
        seen.add(row.call)
        if row.issued > moment:
            raise ValueError("承诺签发时间晚于开班")
        if row.promised <= moment:
            continue
        blocks[row.berth].append((row.moored, row.promised))
    return {berth: _merge(marks) for berth, marks in sorted(blocks.items())}


def issued_records(issued, moment):
    records = []
    for raw in issued:
        row = _as_commitment(raw)
        records.append({"call": row.call, "line": row.line, "berth": row.berth,
                        "moored": row.moored, "promised": row.promised,
                        "issued": row.issued, "kept": row.promised > moment})
    return records


def backtest_slips(raw, cutoff, k=7, min_history=8):
    if type(cutoff) is not int or not 0 <= cutoff <= 30000:
        raise ValueError("校准截止分钟越界")
    if type(k) is not int or k < 1 or type(min_history) is not int or min_history < 1:
        raise ValueError("回测近邻与历史下限须为正整数")
    rows = []
    for entry in raw:
        row = CompletedCall.parse(entry) if isinstance(entry, dict) else entry
        if row.closed < cutoff:
            rows.append(row)
    if len({row.call for row in rows}) != len(rows):
        raise ValueError("校准结案航次重复")
    rows.sort(key=lambda item: (item.closed, item.call))
    slips = defaultdict(list)
    history = []
    index = 0
    while index < len(rows):
        stop = index
        while stop < len(rows) and rows[stop].closed == rows[index].closed:
            stop += 1
        batch = rows[index:stop]
        if len(history) >= min_history:
            model = learn([asdict(item) for item in history], cutoff, k)
            for row in batch:
                guess = predict_dwell(model, row.line, row.boxes, row.reefer,
                                      row.draft, row.cranes, row.yard)
                slips[row.line].append(max(0.0, row.duration - guess.high))
        history.extend(batch)
        index = stop
    return {line: tuple(values) for line, values in sorted(slips.items())}


def nearest_rank(values, p):
    if not values or not 0 < p <= 1:
        raise ValueError("分位口径或样本无效")
    ordered = sorted(values)
    return ordered[max(0, ceil(p * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class RouteBuffer:
    line: str
    support: int
    port_buffer: int
    route_buffer: int | None
    buffer: int
    source: str
    tightened: bool


@dataclass(frozen=True, slots=True)
class Calibration:
    cutoff: int
    port_buffer: int
    points: int
    routes: tuple[RouteBuffer, ...]

    def buffer_for(self, line) -> RouteBuffer:
        for route in self.routes:
            if route.line == line:
                return route
        return RouteBuffer(line, 0, self.port_buffer, None,
                           self.port_buffer, "全港口径", False)


def calibrate(raw, cutoff, min_support=6, p=1.0, k=7, min_history=8,
              default_buffer=DEFAULT_PORT_BUFFER):
    if type(min_support) is not int or min_support < 1:
        raise ValueError("航线收紧样本下限须为正整数")
    if type(default_buffer) is not int or not 0 <= default_buffer <= 4000:
        raise ValueError("全港默认宽放越界")
    slips = backtest_slips(raw, cutoff, k, min_history)
    pooled = [value for values in slips.values() for value in values]
    port = ceil(nearest_rank(pooled, p)) if pooled else default_buffer
    routes = []
    for line, values in slips.items():
        if len(values) >= min_support:
            route = ceil(nearest_rank(values, p))
            buffer = min(port, route)
            routes.append(RouteBuffer(line, len(values), port, route, buffer,
                                      "航线回测", buffer < port))
        else:
            routes.append(RouteBuffer(line, len(values), port, None,
                                      port, "全港口径", False))
    return Calibration(cutoff, port, len(pooled), tuple(routes))


def calibration_records(calibration):
    return [asdict(route) for route in calibration.routes]


@dataclass(frozen=True, slots=True)
class Commitment:
    call: str
    berth: str
    moor: int
    sail: int
    promised: int
    buffer: int
    line: str
    source: str
    support: int


def sign_out(bookings, lines, calibration):
    commits = []
    for order in bookings:
        if order.state != "安排靠泊" or order.berth is None:
            continue
        line = lines.get(order.call)
        if line is None:
            raise ValueError("签出承诺缺少航线")
        route = calibration.buffer_for(line)
        commits.append(Commitment(order.call, order.berth, order.moor, order.sail,
                                  order.sail + route.buffer, route.buffer, line,
                                  route.source, route.support))
    return tuple(sorted(commits, key=lambda item: (item.berth, item.moor, item.call)))


def commitment_digest(commitments, locked, calibration):
    payload = {
        "cutoff": calibration.cutoff,
        "port_buffer": calibration.port_buffer,
        "routes": [asdict(route) for route in calibration.routes],
        "locked": {berth: [list(block) for block in blocks]
                   for berth, blocks in sorted(locked.items())},
        "commitments": [asdict(item) for item in commitments],
    }
    return sha256(dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
