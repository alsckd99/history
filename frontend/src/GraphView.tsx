import { useCallback, useEffect, useState, useRef } from "react";
import ReactFlow, {
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MarkerType,
  useReactFlow,
  ReactFlowProvider,
  Position,
  Handle,
} from "reactflow";
import "reactflow/dist/style.css";
import { fetchKeywords, fetchEntity, preloadVideoKeywords, getPreloadStatus, PreloadStatus } from "./api";

interface GraphViewProps {
  videoId: string;
  currentTime: number;
  onNodeClick: (entityName: string | null, entityData: any) => void;
  selectedNode?: string | null;
  onPlaceNamesExtracted?: (placeNames: string[]) => void;
}

// 깊이별 노드 정보 타입
interface DepthNodeInfo {
  nodeId: string;
  entityName: string;
  depth: number;
  parentId: string | null;
}

// 색상 팔레트
const COLORS = {
  keyword: "#8b5cf6",
  entity: "#3b82f6",
  mapped: "#10b981",
  depth2: "#06b6d4", // 2깊이 노드 (cyan)
  depth3: "#f472b6", // 3깊이 노드 (pink)
  selected: "#f59e0b", // 선택된 노드 색상 (주황색)
  edge: "#64748b",
  edgeActive: "#8b5cf6",
};

// 원형 노드 컴포넌트 - Handle 포함
function CircleNode({ data }: { data: any }) {
  const isKeyword = data.isKeyword;
  const isSelected = data.isSelected;
  const depth = data.depth || 1;
  const size = isKeyword ? 80 : (depth === 1 ? 55 : depth === 2 ? 50 : 45);

  // 깊이에 따른 색상
  const getDepthColor = () => {
    if (isSelected) return COLORS.selected;
    if (isKeyword) return COLORS.keyword;
    if (depth === 1) return COLORS.mapped;
    if (depth === 2) return COLORS.depth2;
    if (depth >= 3) return COLORS.depth3;
    return COLORS.entity;
  };

  const bgColor = getDepthColor();
  const borderColor = isSelected ? "#fbbf24" : isKeyword ? "#a78bfa" : "rgba(255,255,255,0.3)";

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: bgColor,
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: isKeyword ? "11px" : "9px",
        fontWeight: "600",
        textAlign: "center",
        padding: "5px",
        boxShadow: isSelected
          ? "0 0 25px rgba(245, 158, 11, 0.7)"
          : isKeyword
          ? "0 0 20px rgba(139, 92, 246, 0.5)"
          : "0 4px 8px rgba(0,0,0,0.3)",
        border: `3px solid ${borderColor}`,
        cursor: "pointer",
        transition: "all 0.2s ease",
        wordBreak: "keep-all",
        lineHeight: 1.2,
        position: "relative",
      }}
    >
      {/* 연결점 (Handle) - 키워드: 상하좌우, 엔티티: 좌우 */}
      {isKeyword ? (
        <>
          <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
          <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} id="bottom" />
          <Handle type="source" position={Position.Right} style={{ opacity: 0 }} id="right" />
        </>
      ) : (
        <>
          <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
          <Handle type="source" position={Position.Right} style={{ opacity: 0 }} id="right" />
        </>
      )}
      {data.label}
    </div>
  );
}

const nodeTypes = {
  circle: CircleNode,
};

// 내부 그래프 컴포넌트 (useReactFlow 사용을 위해 분리)
function GraphViewInner({ videoId, currentTime, onNodeClick, selectedNode, onPlaceNamesExtracted }: GraphViewProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const prevKeywordsRef = useRef<string[]>([]);
  const nodePositionRef = useRef<{ x: number; y: number }>({ x: 300, y: 200 });
  const lastCenterRef = useRef<{ x: number; y: number } | null>(null);

  const { setCenter, getZoom } = useReactFlow();

  // 특정 위치로 화면 중심 이동
  const focusOnPosition = useCallback((x: number, y: number, zoom?: number) => {
    const currentZoom = zoom || getZoom() || 1;
    setCenter(x, y, { zoom: currentZoom, duration: 500 });
  }, [setCenter, getZoom]);


  // 선택된 노드 표시 업데이트
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          isSelected: node.data.label === selectedNode,
        },
      }))
    );
  }, [selectedNode, setNodes]);

  // 방사형 노드 위치 계산 (중심에서 여러 레벨로 퍼짐)
  const calculateRadialPositions = (
    centerX: number,
    centerY: number,
    nodes: { id: string; level: number; parentAngle?: number }[],
    baseRadius: number = 150
  ) => {
    const positions: { [key: string]: { x: number; y: number } } = {};
    
    // 레벨별로 그룹화
    const levels: { [level: number]: typeof nodes } = {};
    nodes.forEach(node => {
      if (!levels[node.level]) levels[node.level] = [];
      levels[node.level].push(node);
    });
    
    // 각 레벨별로 위치 계산
    Object.entries(levels).forEach(([levelStr, levelNodes]) => {
      const level = parseInt(levelStr);
      const radius = baseRadius * level;
      const angleStep = (2 * Math.PI) / Math.max(levelNodes.length, 1);
      const startAngle = levelNodes[0]?.parentAngle ?? -Math.PI / 2;
      
      levelNodes.forEach((node, index) => {
        // 부모 각도를 기준으로 약간씩 퍼지게
        const spreadAngle = levelNodes.length > 1 
          ? (index - (levelNodes.length - 1) / 2) * (Math.PI / 4) / levelNodes.length
          : 0;
        const angle = node.parentAngle !== undefined 
          ? node.parentAngle + spreadAngle
          : startAngle + angleStep * index;
          
        positions[node.id] = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        };
      });
    });
    
    return positions;
  };

  // 이미 추가된 아이템 추적 (키워드 기반 1깊이만 영구 저장)
  const addedKeywordsRef = useRef<Set<string>>(new Set());
  const addedEntitiesRef = useRef<Set<string>>(new Set()); // 1깊이 엔티티만
  const addedEdgesRef = useRef<Set<string>>(new Set());
  const lastKeywordYRef = useRef<number>(100);
  const lastEntityCountRef = useRef<number>(0); // 마지막 키워드의 엔티티 수 (간격 계산용)
  const prevTimeRef = useRef<number>(0);
  const lastSliceIndexRef = useRef<number>(-1);
  
  // 깊이별 노드 관리 (2깊이 이상은 임시)
  const depthNodesRef = useRef<Map<string, DepthNodeInfo>>(new Map()); // nodeId -> info
  const expandedEntityRef = useRef<string | null>(null); // 현재 확장된 1깊이 엔티티
  
  // 특정 깊이 이상의 노드와 엣지 제거
  const removeNodesFromDepth = useCallback((minDepth: number) => {
    const nodesToRemove: string[] = [];
    
    depthNodesRef.current.forEach((info, nodeId) => {
      if (info.depth >= minDepth) {
        nodesToRemove.push(nodeId);
      }
    });
    
    const edgesToRemove: string[] = [];
    
    // 해당 노드들과 연결된 엣지도 제거
    setEdges((eds) => eds.filter((edge) => {
      const shouldRemove = nodesToRemove.includes(edge.source) || nodesToRemove.includes(edge.target);
      if (shouldRemove) {
        edgesToRemove.push(edge.id);
      }
      return !shouldRemove;
    }));
    
    // 노드 제거
    setNodes((nds) => nds.filter((node) => !nodesToRemove.includes(node.id)));
    
    // ref에서도 제거 (노드와 엣지 캐시 모두)
    nodesToRemove.forEach((nodeId) => {
      depthNodesRef.current.delete(nodeId);
    });
    edgesToRemove.forEach((edgeId) => {
      addedEdgesRef.current.delete(edgeId);
    });
    
    console.log(`[GraphView] 깊이 ${minDepth} 이상 노드 제거:`, nodesToRemove);
  }, [setNodes, setEdges]);
  
  // 특정 노드에서 파생된 자식 노드들만 제거
  const removeChildNodes = useCallback((parentNodeId: string) => {
    const nodesToRemove: string[] = [];
    const edgesToRemove: string[] = [];
    
    // 재귀적으로 자식 노드 찾기
    const findChildren = (parentId: string) => {
      depthNodesRef.current.forEach((info, nodeId) => {
        if (info.parentId === parentId) {
          nodesToRemove.push(nodeId);
          findChildren(nodeId); // 자식의 자식도 찾기
        }
      });
    };
    
    findChildren(parentNodeId);
    
    if (nodesToRemove.length === 0) return;
    
    // 해당 노드들과 연결된 엣지도 제거
    setEdges((eds) => eds.filter((edge) => {
      const shouldRemove = nodesToRemove.includes(edge.source) || nodesToRemove.includes(edge.target);
      if (shouldRemove) {
        edgesToRemove.push(edge.id);
      }
      return !shouldRemove;
    }));
    
    // 노드 제거
    setNodes((nds) => nds.filter((node) => !nodesToRemove.includes(node.id)));
    
    // ref에서도 제거 (노드와 엣지 캐시 모두)
    nodesToRemove.forEach((nodeId) => {
      depthNodesRef.current.delete(nodeId);
    });
    edgesToRemove.forEach((edgeId) => {
      addedEdgesRef.current.delete(edgeId);
    });
    
    console.log(`[GraphView] '${parentNodeId}'의 자식 노드 제거:`, nodesToRemove);
  }, [setNodes, setEdges]);
  
  // 현재 확장된 2깊이 노드 추적
  const expanded2DepthRef = useRef<string | null>(null);
  
  // 엔티티 클릭 시 하위 노드 추가
  const expandEntity = useCallback(async (entityName: string, parentNodeId: string, parentDepth: number, parentX: number, parentY: number) => {
    const newDepth = parentDepth + 1;
    
    // 최대 깊이 제한 (3까지)
    if (newDepth > 3) {
      console.log(`[GraphView] 최대 깊이(3) 도달, 확장 중지`);
      return;
    }
    
    try {
      console.log(`[GraphView] 엔티티 '${entityName}' 확장 (깊이 ${newDepth})`);
      const data = await fetchEntity(entityName, 1);
      
      console.log(`[GraphView] API 응답:`, data.neighbors);
      
      // neighbors.entities 배열에서 실제 엔티티 추출
      const neighbors = data.neighbors as any;
      const neighborEntities: any[] = neighbors?.entities || [];
      
      if (!neighborEntities || neighborEntities.length === 0) {
        console.log(`[GraphView] '${entityName}'의 이웃 엔티티 없음`);
        return;
      }
      
      // 엔티티 이름 추출 (name 또는 _key 필드 사용)
      const neighborNames: string[] = neighborEntities
        .map((ent: any) => ent.name || ent._key || '')
        .filter((name: string) => name && name !== entityName)
        .slice(0, 5); // 최대 5개
      
      console.log(`[GraphView] 이웃 엔티티 이름:`, neighborNames);
      
      if (neighborNames.length === 0) {
        console.log(`[GraphView] 유효한 이웃 없음`);
        return;
      }
      
      // 먼저 유효한 이웃만 필터링 (중복 제거)
      const validNeighbors = neighborNames.filter((neighborName: string) => {
        const depthNodeId = `depth${newDepth}-${neighborName}`;
        // 키워드나 1깊이 엔티티와 중복되면 스킵
        if (addedKeywordsRef.current.has(neighborName) || addedEntitiesRef.current.has(neighborName)) {
          return false;
        }
        // 이미 같은 깊이에 추가된 노드면 스킵
        if (depthNodesRef.current.has(depthNodeId)) {
          return false;
        }
        return true;
      });
      
      if (validNeighbors.length === 0) {
        console.log(`[GraphView] 유효한 이웃 없음 (중복 제외 후)`);
        return;
      }
      
      const entityVerticalGap = 55;
      const horizontalOffset = 150;
      // 유효한 노드 수 기준으로 중앙 정렬
      const totalHeight = (validNeighbors.length - 1) * entityVerticalGap;
      const startY = parentY - totalHeight / 2;
      
      const newNodes: Node[] = [];
      const newEdges: Edge[] = [];
      
      validNeighbors.forEach((neighborName: string, idx: number) => {
        const depthNodeId = `depth${newDepth}-${neighborName}`;
        
        const nodePos = {
          x: parentX + horizontalOffset,
          y: startY + idx * entityVerticalGap,
        };
        
        newNodes.push({
          id: depthNodeId,
          type: "circle",
          position: nodePos,
          data: {
            label: neighborName,
            isKeyword: false,
            isMapped: false,
            depth: newDepth,
            parentId: parentNodeId,
            isSelected: neighborName === selectedNode,
          },
        });
        
        // 깊이 정보 저장
        depthNodesRef.current.set(depthNodeId, {
          nodeId: depthNodeId,
          entityName: neighborName,
          depth: newDepth,
          parentId: parentNodeId,
        });
        
        // 부모 → 자식 연결 (중복 체크)
        const edgeId = `edge-${parentNodeId}-${depthNodeId}`;
        if (!addedEdgesRef.current.has(edgeId)) {
          newEdges.push({
            id: edgeId,
            source: parentNodeId,
            sourceHandle: "right",
            target: depthNodeId,
            type: "default",
            style: { stroke: newDepth === 2 ? COLORS.depth2 : COLORS.depth3, strokeWidth: 2 },
          });
          addedEdgesRef.current.add(edgeId);
        }
      });
      
      if (newNodes.length > 0) {
        setNodes((prev) => [...prev, ...newNodes]);
        setEdges((prev) => {
          // 기존 엣지와 중복되지 않는 새 엣지만 추가
          const existingIds = new Set(prev.map(e => e.id));
          const uniqueNewEdges = newEdges.filter(e => !existingIds.has(e.id));
          return [...prev, ...uniqueNewEdges];
        });
        console.log(`[GraphView] 깊이 ${newDepth} 노드 ${newNodes.length}개 추가`);
      }
    } catch (err) {
      console.error(`[GraphView] 엔티티 확장 오류:`, err);
    }
  }, [selectedNode, setNodes, setEdges]);
  
  // 노드 클릭 핸들러 수정 - 깊이별 확장/축소
  const handleNodeClickWithExpand = useCallback(
    async (_: any, node: Node) => {
      const clickedName = node.data.label;
      const clickedDepth = node.data.depth || (node.data.isKeyword ? 0 : 1);
      const nodeId = node.id;
      
      // 같은 노드 다시 클릭 → 선택 해제 + 자식 노드 삭제
      if (selectedNode === clickedName) {
        // 이 노드에서 파생된 자식 노드들 제거
        removeChildNodes(nodeId);
        
        // 확장 상태 업데이트
        if (clickedDepth === 1) {
          expandedEntityRef.current = null;
          expanded2DepthRef.current = null;
        } else if (clickedDepth === 2) {
          expanded2DepthRef.current = null;
        }
        
        onNodeClick(null, null);
        return;
      }
      
      // 화면 이동
      const nodeSize = node.data.isKeyword ? 40 : 27;
      focusOnPosition(node.position.x + nodeSize, node.position.y + nodeSize);
      onNodeClick(clickedName, node.data);
      
      // 키워드 노드는 확장하지 않음
      if (node.data.isKeyword) {
        return;
      }
      
      // 1깊이 엔티티 클릭 시
      if (clickedDepth === 1) {
        // 다른 1깊이 엔티티가 이미 확장되어 있으면 그 하위 노드들 제거
        if (expandedEntityRef.current && expandedEntityRef.current !== clickedName) {
          console.log(`[GraphView] 이전 확장 '${expandedEntityRef.current}' 축소`);
          removeNodesFromDepth(2);
          expanded2DepthRef.current = null;
        }
        
        // 새 1깊이 엔티티 확장
        expandedEntityRef.current = clickedName;
        await expandEntity(clickedName, nodeId, 1, node.position.x, node.position.y);
      }
      // 2깊이 노드 클릭 시 3깊이 확장
      else if (clickedDepth === 2) {
        // 다른 2깊이 노드가 이미 확장되어 있으면 그 자식들만 제거
        if (expanded2DepthRef.current && expanded2DepthRef.current !== nodeId) {
          removeChildNodes(expanded2DepthRef.current);
        }
        
        // 새 2깊이 노드 확장
        expanded2DepthRef.current = nodeId;
        await expandEntity(clickedName, nodeId, 2, node.position.x, node.position.y);
      }
      // 3깊이 노드는 더 이상 확장하지 않음
    },
    [onNodeClick, selectedNode, focusOnPosition, removeNodesFromDepth, removeChildNodes, expandEntity]
  );
  
  // 프리로드 상태
  const [preloadStatus, setPreloadStatus] = useState<PreloadStatus | null>(null);
  const preloadStartedRef = useRef<boolean>(false);

  // 재생 위치가 크게 뒤로 이동하면 리셋 (15초 이상 뒤로 가면)
  useEffect(() => {
    if (!videoId) return;
    if (currentTime < prevTimeRef.current - 15) {
      console.log(`[GraphView] 되감기 감지: ${prevTimeRef.current} -> ${currentTime}, 그래프 초기화`);
      setNodes([]);
      setEdges([]);
      addedKeywordsRef.current.clear();
      addedEntitiesRef.current.clear();
      addedEdgesRef.current.clear();
      lastKeywordYRef.current = 100;
      lastEntityCountRef.current = 0;
      lastSliceIndexRef.current = -1;
      depthNodesRef.current.clear();
      expandedEntityRef.current = null;
    }
    prevTimeRef.current = currentTime;
  }, [currentTime, videoId, setEdges, setNodes]);

  useEffect(() => {
    if (!videoId) return;

    const loadKeywords = async () => {
      try {
        // 현재 시간을 기준으로 해당 시점의 키워드 요청
        const currentSec = Math.max(0, Math.floor(currentTime));
        
        // 15초 단위로 슬라이스 인덱스 계산
        const sliceDuration = 15;
        const sliceIndex = Math.floor(currentSec / sliceDuration);

        // 같은 슬라이스면 스킵
        if (sliceIndex === lastSliceIndexRef.current) {
          return;
        }
        
        console.log(`[GraphView] 슬라이스 변경: ${lastSliceIndexRef.current} -> ${sliceIndex} (currentTime: ${currentSec}초)`);
        lastSliceIndexRef.current = sliceIndex;

        // 해당 슬라이스의 시작~끝 시간으로 요청
        const start = sliceIndex * sliceDuration;
        const end = start + sliceDuration;

        console.log(`[GraphView] API 호출: start=${start}, end=${end}`);
        const data = await fetchKeywords(videoId, start, end);

        console.log(`[GraphView] API 응답:`, data.keywords?.map((k: any) => k.term));

        if (!data.keywords || data.keywords.length === 0) {
          console.log(`[GraphView] 키워드 없음, 스킵`);
          return;
        }

        // 새로 추가할 키워드만 필터링
        const newKeywords = data.keywords.filter(
          (k: any) => !addedKeywordsRef.current.has(k.term)
        );

        // 새 키워드가 없으면 스킵
        if (newKeywords.length === 0) {
          return;
        }

        console.log(`[GraphView] 새 키워드 추가:`, newKeywords.map((k: any) => k.term));

        const centerX = 150; // 키워드는 왼쪽에 배치
        const entityVerticalGap = 65; // 엔티티 간 세로 간격
        const minKeywordSpacing = 120; // 최소 키워드 간격
        const keywordPadding = 40; // 키워드 위아래 여백

        // 새 엔티티 추가 (mapped_entities에서)
        const allEntities = data.mapped_entities || [];
        console.log(`[GraphView] mapped_entities:`, allEntities.map((e: any) => e.name));
        
        const newEntities = allEntities.filter(
          (e: any) => e.name && !addedEntitiesRef.current.has(e.name) && !addedKeywordsRef.current.has(e.name)
        );
        console.log(`[GraphView] 새 엔티티:`, newEntities.map((e: any) => e.name));

        // 각 키워드에 배분될 엔티티 수 미리 계산
        const entitiesPerKeyword = Math.max(1, Math.ceil(newEntities.length / Math.max(1, newKeywords.length)));

        const newNodes: Node[] = [];
        const newEdges: Edge[] = [];
        let currentY = lastKeywordYRef.current;
        let prevEntityCount = lastEntityCountRef.current; // 이전 슬라이스의 마지막 키워드 엔티티 수

        // 새 키워드 노드 추가 (세로로 배치, 엔티티 수에 따라 간격 조정)
        newKeywords.forEach((keyword: any, kwIdx: number) => {
          const keywordId = `keyword-${keyword.term}`;
          
          // 이 키워드에 할당될 엔티티 계산
          const startIdx = kwIdx * entitiesPerKeyword;
          const endIdx = Math.min(startIdx + entitiesPerKeyword, newEntities.length);
          const keywordEntities = newEntities.slice(startIdx, endIdx);
          const entityCount = keywordEntities.length;
          
          // 엔티티 수에 따른 동적 간격 계산 (이전 + 현재 키워드의 엔티티 블록 높이 모두 고려)
          const prevBlockHalfHeight = Math.max(0, (prevEntityCount - 1)) * entityVerticalGap / 2;
          const currBlockHalfHeight = Math.max(0, (entityCount - 1)) * entityVerticalGap / 2;
          const requiredSpacing = Math.max(minKeywordSpacing, prevBlockHalfHeight + currBlockHalfHeight + keywordPadding * 2);
          
          // 첫 번째 키워드가 아니면 간격 추가
          if (kwIdx > 0 || addedKeywordsRef.current.size > 0) {
            currentY += requiredSpacing;
          }
          
          const keywordY = currentY;
          prevEntityCount = entityCount; // 다음 반복을 위해 저장
          
          const prevKeywordsList = Array.from(addedKeywordsRef.current);
          const prevKeywordId =
            kwIdx > 0
              ? `keyword-${newKeywords[kwIdx - 1].term}`
              : (prevKeywordsList.length > 0 ? `keyword-${prevKeywordsList[prevKeywordsList.length - 1]}` : null);

          newNodes.push({
            id: keywordId,
            type: "circle",
            position: { x: centerX, y: keywordY },
            sourcePosition: Position.Right,
            targetPosition: Position.Left,
            data: {
              label: keyword.term,
              score: keyword.score,
              isKeyword: true,
              isSelected: keyword.term === selectedNode,
            },
          });

          // 이전 키워드와 연결 (체인 형태 - 부드러운 베지어 곡선)
          if (prevKeywordId) {
            const edgeId = `edge-${prevKeywordId}-${keywordId}`;
            if (!addedEdgesRef.current.has(edgeId)) {
              newEdges.push({
                id: edgeId,
                source: prevKeywordId,
                sourceHandle: "bottom",
                target: keywordId,
                type: "default", // 베지어 곡선 (더 부드러움)
                style: { stroke: COLORS.edgeActive, strokeWidth: 2 },
                animated: true,
              });
              addedEdgesRef.current.add(edgeId);
            }
          }

          addedKeywordsRef.current.add(keyword.term);

          // 엔티티를 오른쪽에 세로로 배치
          if (entityCount > 0) {
            const entityStartX = centerX + 180; // 키워드에서 오른쪽으로 떨어진 거리
            const totalEntityHeight = (entityCount - 1) * entityVerticalGap;
            const entityStartY = keywordY - totalEntityHeight / 2; // 키워드 중심 기준으로 세로 중앙 정렬

            console.log(`[GraphView] 키워드 '${keyword.term}'에 엔티티 배분 (${entityCount}개):`, keywordEntities.map((e: any) => e.name));

            keywordEntities.forEach((entity: any, entIdx: number) => {
              const entityId = `entity-${entity.name}`;
              
              if (!addedEntitiesRef.current.has(entity.name)) {
                const entityPos = {
                  x: entityStartX,
                  y: entityStartY + entIdx * entityVerticalGap,
                };

                newNodes.push({
                  id: entityId,
                  type: "circle",
                  position: entityPos,
                  data: {
                    label: entity.name,
                    isKeyword: false,
                    isMapped: true,
                    depth: 1, // 1깊이 엔티티
                    isSelected: entity.name === selectedNode,
                  },
                });

                addedEntitiesRef.current.add(entity.name);

                // 키워드 → 엔티티 연결 (부드러운 베지어 곡선)
                const edgeId = `edge-${keyword.term}-${entity.name}`;
                if (!addedEdgesRef.current.has(edgeId)) {
                  newEdges.push({
                    id: edgeId,
                    source: keywordId,
                    sourceHandle: "right",
                    target: entityId,
                    type: "default", // 베지어 곡선
                    style: { stroke: COLORS.edge, strokeWidth: 2 },
                  });
                  addedEdgesRef.current.add(edgeId);
                }
              }
            });
          }
        });

        // 마지막 키워드 Y 위치 업데이트 (다음 슬라이스의 키워드를 위해)
        // prevEntityCount에는 마지막 키워드의 엔티티 수가 저장되어 있음
        lastKeywordYRef.current = currentY;
        // 마지막 키워드의 엔티티 블록 절반 높이를 저장 (다음 키워드와의 간격 계산에 사용)
        lastEntityCountRef.current = prevEntityCount;
        
        console.log(`[GraphView] 생성된 노드:`, newNodes.length, `엣지:`, newEdges.length);

        // 새 노드/엣지만 추가 (기존 것 유지)
        if (newNodes.length > 0) {
          setNodes(prev => [...prev, ...newNodes]);
          setEdges(prev => [...prev, ...newEdges]);

          // 새 키워드 노드 위치로 화면 이동
          setTimeout(() => {
            focusOnPosition(centerX + 100, currentY, 0.9);
          }, 150);
        }

        // 지명 추출 및 콜백 호출 (지도 연동)
        if (onPlaceNamesExtracted) {
          // 지명 관련 타입: location, 지명, 장소, 지역, 도시, 행정 등
          const placeTypes = ['location', '지명', '장소', '지역', '도시', '행정', '산지', '하천', '섬'];
          
          // mapped_entities에서 지명 타입 추출
          const placeEntities = allEntities
            .filter((e: any) => {
              const entityType = (e.type || '').toLowerCase();
              return placeTypes.some(pt => entityType.includes(pt));
            })
            .map((e: any) => e.name);

          // 키워드 중 지명으로 추정되는 것들도 포함
          // (예: "한양", "한산도", "부산" 등 - 지명 관련 키워드)
          const placeKeywords = newKeywords
            .filter((k: any) => {
              const term = k.term || '';
              // 지명 관련 접미사나 패턴 체크
              return /[도시군구읍면리동]$/.test(term) || 
                     /[산강해섬포진]$/.test(term) ||
                     term.includes('도') || term.includes('성');
            })
            .map((k: any) => k.term);

          // 모든 키워드도 지도 검색 대상에 포함 (옛 지명 DB에서 찾을 수 있으므로)
          const allPlaceNames = [...new Set([
            ...placeEntities,
            ...placeKeywords,
            ...newKeywords.map((k: any) => k.term) // 모든 키워드 포함
          ])];

          if (allPlaceNames.length > 0) {
            console.log(`[GraphView] 지명 추출:`, allPlaceNames);
            onPlaceNamesExtracted(allPlaceNames);
          }
        }
      } catch (err: any) {
        console.error("키워드 로드 오류:", err);
        setError(err.message || "키워드를 불러올 수 없습니다");
      } finally {
        setLoading(false);
      }
    };

    loadKeywords();
  }, [videoId, currentTime, focusOnPosition]);

  // 그래프 초기화 (영상 변경 시)
  useEffect(() => {
    setNodes([]);
    setEdges([]);
    prevKeywordsRef.current = [];
    nodePositionRef.current = { x: 300, y: 200 };
    addedKeywordsRef.current = new Set();
    addedEntitiesRef.current = new Set();
    lastKeywordYRef.current = 100;
    lastEntityCountRef.current = 0;
    addedEdgesRef.current = new Set();
    lastSliceIndexRef.current = -1;
    depthNodesRef.current.clear();
    expandedEntityRef.current = null;
    preloadStartedRef.current = false;
    setPreloadStatus(null);
  }, [videoId]);

  // 영상 시작 시 프리로드 시작 (백그라운드에서 순차 처리)
  useEffect(() => {
    if (!videoId || preloadStartedRef.current) return;
    
    preloadStartedRef.current = true;
    console.log(`[GraphView] 프리로드 시작: ${videoId}`);
    
    // 프리로드 시작
    preloadVideoKeywords(videoId, 5)
      .then((status) => {
        console.log(`[GraphView] 프리로드 상태:`, status);
        setPreloadStatus(status);
      })
      .catch((err) => {
        console.error(`[GraphView] 프리로드 오류:`, err);
      });
    
    // 주기적으로 상태 확인 (2초마다)
    const statusInterval = setInterval(() => {
      getPreloadStatus(videoId)
        .then((status) => {
          setPreloadStatus(status);
          if (status.status === "complete") {
            console.log(`[GraphView] 프리로드 완료!`);
            clearInterval(statusInterval);
          }
        })
        .catch(() => {});
    }, 2000);
    
    return () => clearInterval(statusInterval);
  }, [videoId]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#0f172a",
        borderRadius: "8px",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* 프리로드 상태 표시 */}
      {preloadStatus && preloadStatus.is_loading && (
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            padding: "6px 12px",
            color: "#a78bfa",
            background: "rgba(30, 41, 59, 0.95)",
            borderRadius: "12px",
            zIndex: 10,
            fontSize: "11px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "#a78bfa",
              animation: "pulse 1s infinite",
            }}
          />
          <span>
            노드 준비 중 {preloadStatus.loaded_slices}/{preloadStatus.total_slices}
          </span>
        </div>
      )}
      {loading && (
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            padding: "4px 10px",
            color: "#60a5fa",
            background: "rgba(30, 41, 59, 0.9)",
            borderRadius: "12px",
            zIndex: 10,
            fontSize: "12px",
          }}
        >
          로딩 중...
        </div>
      )}
      {error && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            color: "#f87171",
            fontSize: "13px",
            textAlign: "center",
          }}
        >
          {error}
        </div>
      )}
      {nodes.length === 0 && !loading && !error && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#0f172a",
            zIndex: 5,
          }}
        >
          <span style={{ color: "#6b7280", fontSize: "13px" }}>
            영상을 업로드하면 그래프가 표시됩니다
          </span>
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClickWithExpand}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={{
          type: "default", // 베지어 곡선 (가장 부드러움)
          style: { strokeWidth: 2, stroke: COLORS.edge },
        }}
        fitView={nodes.length === 0}
        fitViewOptions={{ padding: 0.5 }}
        style={{ background: "#0f172a" }}
        minZoom={0.2}
        maxZoom={2}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background color="#1e293b" gap={20} />
        <Controls
          style={{
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
          }}
        />
      </ReactFlow>
    </div>
  );
}

// ReactFlowProvider로 감싸는 Wrapper 컴포넌트
export default function GraphView(props: GraphViewProps) {
  return (
    <ReactFlowProvider>
      <GraphViewInner {...props} />
    </ReactFlowProvider>
  );
}
