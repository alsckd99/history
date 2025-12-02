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

// 색상 팔레트 - 카테고리별 색상
const COLORS = {
  keyword: "#8b5cf6",      // 주제 (보라)
  entity: "#3b82f6",       // 기본 엔티티
  // 카테고리별 색상 (인물, 세력, 지명)
  person: "#f472b6",       // 인물 (핑크)
  faction: "#3b82f6",      // 세력 (파랑)
  place: "#10b981",        // 지명 (초록)
  selected: "#fbbf24",     // 선택된 노드 (노랑)
  edge: "#64748b",
  edgeActive: "#8b5cf6",
  edgePerson: "#f472b6",   // 인물 연결선
  edgeFaction: "#3b82f6",  // 세력 연결선 (파랑)
  edgePlace: "#10b981",    // 지명 연결선
};

// 카테고리 분류 함수
const getCategoryType = (category: string, type: string): 'person' | 'faction' | 'place' | null => {
  const cat = (category || '').toLowerCase();
  const t = (type || '').toLowerCase();
  
  // 인물 판별
  if (t === '인물' || cat.includes('인물') || 
      cat.includes('유교') || cat.includes('불교') || 
      cat.includes('종교') || cat.includes('철학')) {
    return 'person';
  }
  
  // 세력 판별
  if (t === '세력') {
    return 'faction';
  }
  
  // 지명 판별 (지명, 전투지역 포함)
  if (t === '지명' || t === '전투지역' || cat.includes('지명') || 
      cat.includes('지리/인문지리') || cat.includes('지리/자연지리') ||
      cat.includes('지리')) {
    return 'place';
  }
  
  return null;
};

// 깊이에 따른 노드 크기 계산 함수 (전역)
const getNodeSizeByDepth = (depth: number, isKeyword: boolean = false): number => {
  if (isKeyword) return 100;
  // 1깊이: 80, 2깊이: 75, 3깊이: 70, 4깊이: 65, ... (5씩 감소, 최소 50)
  const size = 85 - (depth * 5);
  return Math.max(size, 50);
};

// 원형 노드 컴포넌트 - Handle 포함
function CircleNode({ data }: { data: any }) {
  const isKeyword = data.isKeyword;
  const isSelected = data.isSelected;
  const isHighlighted = data.isHighlighted;  // 선택된 노드 또는 부모 노드
  const isFaded = data.isFaded;  // 페이드아웃 대상
  const categoryType = data.categoryType as 'person' | 'faction' | 'place' | null;
  const depth = data.depth || 0;
  
  // 깊이에 따른 노드 크기 (동적 계산)
  const baseSize = getNodeSizeByDepth(depth, isKeyword);
  // 선택 시 transform scale로 확대 (위치 유지)
  const scale = isSelected ? 1.2 : 1;

  // 카테고리별 색상 (선택 시 색상 유지)
  const getNodeColor = () => {
    if (isKeyword) return COLORS.keyword;
    if (categoryType === 'person') return COLORS.person;
    if (categoryType === 'faction') return COLORS.faction;
    if (categoryType === 'place') return COLORS.place;
    return COLORS.entity;
  };

  const bgColor = getNodeColor();
  
  // 테두리 색상 (선택 시에도 엔티티 색상 유지, 밝게)
  const getBorderColor = () => {
    if (isKeyword) return "#a78bfa";
    if (categoryType === 'person') return "#f9a8d4";  // 핑크 밝게
    if (categoryType === 'faction') return "#93c5fd"; // 파랑 밝게
    if (categoryType === 'place') return "#6ee7b7";   // 초록 밝게
    return "rgba(255,255,255,0.6)";
  };
  const borderColor = getBorderColor();

  // 글씨 크기 (깊이에 따라 동적 조절)
  const getFontSize = () => {
    if (isKeyword) return 24;
    // 1깊이: 18, 2깊이: 16, 3깊이: 14, 4깊이: 12, ... (2씩 감소, 최소 10)
    const size = 20 - (depth * 2);
    return Math.max(size, 10);
  };
  const fontSize = getFontSize();

  // 선택 시 그림자 색상도 엔티티 색상에 맞춤
  const getSelectedShadow = () => {
    if (isKeyword) return "0 0 30px rgba(139, 92, 246, 0.8)";
    if (categoryType === 'person') return "0 0 30px rgba(244, 114, 182, 0.8)";
    if (categoryType === 'faction') return "0 0 30px rgba(59, 130, 246, 0.8)";
    if (categoryType === 'place') return "0 0 30px rgba(16, 185, 129, 0.8)";
    return "0 0 30px rgba(59, 130, 246, 0.8)";
  };

  // 페이드아웃 시 투명도
  const opacity = isFaded ? 0.3 : 1;

  return (
    <div
      style={{
        width: baseSize,
        height: baseSize,
        borderRadius: "50%",
        background: bgColor,
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: `${fontSize}px`,
        fontWeight: isSelected ? "800" : "600",
        textAlign: "center",
        padding: "8px",
        boxShadow: isSelected
          ? getSelectedShadow()
          : isKeyword
          ? "0 0 20px rgba(139, 92, 246, 0.5)"
          : "0 4px 10px rgba(0,0,0,0.4)",
        border: isSelected ? `4px solid ${borderColor}` : `3px solid ${borderColor}`,
        cursor: "pointer",
        transition: "all 0.3s ease-out, transform 0.2s ease, opacity 0.3s ease",
        transform: `scale(${scale})`,
        opacity: opacity,
        wordBreak: "keep-all",
        lineHeight: 1.2,
        position: "relative",
      }}
    >
      {/* 연결점 (Handle) - 아래쪽으로 뻗어나가는 구조 */}
      {isKeyword ? (
        <>
          <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
          <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} id="bottom" />
        </>
      ) : (
        <>
          <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
          <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} id="bottom" />
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


  // 선택된 노드와 부모 노드 하이라이트, 나머지 페이드아웃
  useEffect(() => {
    if (!selectedNode) {
      // 선택 해제 시 모든 노드 원래 상태로
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
            isSelected: false,
            isHighlighted: false,
            isFaded: false,
        },
      }))
    );
      // 엣지도 원래 상태로
      setEdges((eds) =>
        eds.map((edge) => ({
          ...edge,
          style: {
            ...edge.style,
            stroke: edge.id.startsWith('relation-') ? '#fbbf24' : COLORS.edge,
            opacity: 1,
          },
          animated: edge.id.startsWith('relation-'),
        }))
      );
      return;
    }

    // 선택된 노드 찾기 (term 또는 label로 비교)
    const selectedNodeObj = nodes.find(n => 
      (n.data.term || n.data.label) === selectedNode || n.data.label === selectedNode
    );
    if (!selectedNodeObj) return;

    // 선택된 노드의 부모 ID 찾기 (1깊이 위까지)
    const parentId = selectedNodeObj.data.parentId;
    const highlightedIds = new Set<string>([selectedNodeObj.id]);
    if (parentId) {
      highlightedIds.add(parentId);
    }

    // 노드 업데이트: 선택/부모는 하이라이트, 나머지는 페이드아웃
    setNodes((nds) =>
      nds.map((node) => {
        // term이 있으면 term으로, 없으면 label로 비교
        const nodeTerm = node.data.term || node.data.label;
        const isSelected = nodeTerm === selectedNode;
        const isHighlighted = highlightedIds.has(node.id);
        const isFaded = !isHighlighted;

        return {
          ...node,
          data: {
            ...node.data,
            isSelected,
            isHighlighted,
            isFaded,
          },
        };
      })
    );

    // 엣지 업데이트: 선택된 노드와 부모 사이의 엣지만 강조
    setEdges((eds) =>
      eds.map((edge) => {
        // relation 엣지는 별도 처리
        if (edge.id.startsWith('relation-')) {
          const isHighlighted = highlightedIds.has(edge.source) && highlightedIds.has(edge.target);
          return {
            ...edge,
            style: {
              ...edge.style,
              opacity: isHighlighted ? 1 : 0.2,
            },
            animated: isHighlighted,
          };
        }

        // 일반 엣지: 선택된 노드와 부모 사이인지 확인
        const isHighlightedEdge = 
          (edge.source === parentId && edge.target === selectedNodeObj.id) ||
          (edge.target === parentId && edge.source === selectedNodeObj.id);

        if (isHighlightedEdge) {
          // 선택된 노드의 색상으로 엣지 색상 변경
          const categoryType = selectedNodeObj.data.categoryType;
          const edgeColor = categoryType === 'person' ? COLORS.edgePerson 
                          : categoryType === 'faction' ? COLORS.edgeFaction 
                          : categoryType === 'place' ? COLORS.edgePlace
                          : COLORS.edge;
          return {
            ...edge,
            style: {
              ...edge.style,
              stroke: edgeColor,
              strokeWidth: 3,
              opacity: 1,
            },
          };
        }

        // 나머지 엣지는 페이드아웃
        return {
          ...edge,
          style: {
            ...edge.style,
            stroke: COLORS.edge,
            strokeWidth: 2,
            opacity: 0.2,
          },
        };
      })
    );
  }, [selectedNode, nodes, setNodes, setEdges]);

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
  
  // 수동 children 데이터 저장 (JSON에서 가져온 것)
  const childrenDataRef = useRef<Map<string, any[]>>(new Map());
  
  // relations 데이터 저장 (시간에 따른 키워드 연결)
  const relationsDataRef = useRef<any[]>([]);
  const addedRelationsRef = useRef<Set<string>>(new Set()); // 이미 추가된 relation 추적
  
  // 현재 시간 ref (expandEntity에서 사용)
  const currentTimeRef = useRef<number>(0);
  useEffect(() => {
    currentTimeRef.current = currentTime;
  }, [currentTime]);
  
  // 엔티티/키워드 클릭 시 하위 노드 추가 (아래쪽으로 뻗어나감)
  const expandEntity = useCallback(async (entityName: string, parentNodeId: string, parentDepth: number, parentX: number, parentY: number) => {
    const newDepth = parentDepth + 1;
    
    // 최대 깊이 제한 (3까지)
    if (newDepth > 3) {
      console.log(`[GraphView] 최대 깊이(3) 도달, 확장 중지`);
      return;
    }
    
    try {
      console.log(`[GraphView] '${entityName}' 확장 (깊이 ${newDepth}), 현재 시간: ${currentTimeRef.current}초`);
      
      // 1. JSON에 정의된 children 확인 (수동 노드)
      const manualChildren = childrenDataRef.current.get(entityName) || [];
      
      let childEntities: any[] = [];
      
      if (manualChildren.length > 0) {
        // 수동 정의된 children 사용 - 현재 시간에 맞는 것만 필터링
        const currentSec = currentTimeRef.current;
        const filteredChildren = manualChildren.filter((child: any) => {
          // 시간 정보가 없으면 항상 표시
          if (child.start === undefined || child.end === undefined) return true;
          // 현재 시간이 child의 시간 범위 내에 있는지 확인
          return currentSec >= child.start && currentSec <= child.end;
        });
        
        console.log(`[GraphView] 수동 children (시간 필터링):`, filteredChildren.map((c: any) => `${c.term}(${c.start}-${c.end})`));
        
        childEntities = filteredChildren.map((child: any) => ({
          name: child.term,
          start: child.start,
          end: child.end,
          children: child.children || [],
          isManual: true,
        }));
        
        // 자식의 children도 저장
        filteredChildren.forEach((child: any) => {
          if (child.children && child.children.length > 0) {
            childrenDataRef.current.set(child.term, child.children);
          }
        });
      } else {
        // 2. 수동 children이 없으면 API에서 엔티티 정보 가져오기
        const data = await fetchEntity(entityName, 1);
        console.log(`[GraphView] API 응답:`, data);
        
        // GraphDB에 있으면 neighbors 사용
        if (data.entity) {
          const neighbors = data.neighbors as any;
          const neighborEntities: any[] = neighbors?.entities || [];
          
          // 카테고리별로 분류 (인물, 사건, 지명만 필터링)
          neighborEntities.forEach((ent: any) => {
            const name = ent.name || ent._key || '';
            if (!name || name === entityName) return;
            if (childEntities.length >= 5) return;
            
            const category = ent.category || '';
            const type = ent.type || '';
            const catType = ent._category_type || getCategoryType(category, type);
            
            if (catType) {
              childEntities.push({
                name,
                categoryType: catType,
                isManual: false,
              });
            }
          });
        }
      }
      
      if (childEntities.length === 0) {
        console.log(`[GraphView] '${entityName}'의 하위 노드 없음`);
          return;
        }

      const newNodes: Node[] = [];
      const newEdges: Edge[] = [];
      
      // 아래쪽으로 뻗어나가는 레이아웃
      const verticalOffset = 120;    // 부모와의 수직 거리
      const horizontalGap = 120;     // 형제 노드 간 수평 간격
      
      // 전체 너비 계산 (자식 노드들을 가로로 배치)
      const totalWidth = (childEntities.length - 1) * horizontalGap;
      const startX = parentX - totalWidth / 2;
      
      childEntities.forEach((ent: any, idx: number) => {
        const childName = ent.name;
        
        // 중복 체크
        if (addedKeywordsRef.current.has(childName) || addedEntitiesRef.current.has(childName)) return;
        const depthNodeId = `depth${newDepth}-${childName}`;
        if (depthNodesRef.current.has(depthNodeId)) return;
        
        const nodePos = {
          x: startX + idx * horizontalGap,
          y: parentY + verticalOffset,
        };
        
        // 카테고리 타입 결정
        const catType = ent.categoryType || null;
        const edgeColor = catType === 'person' ? COLORS.edgePerson 
                        : catType === 'faction' ? COLORS.edgeFaction 
                        : catType === 'place' ? COLORS.edgePlace
                        : COLORS.edge;
        
        newNodes.push({
          id: depthNodeId,
          type: "circle",
          position: nodePos,
          data: {
            label: childName,
            isKeyword: false,
            categoryType: catType,
            depth: newDepth,
            parentId: parentNodeId,
            isSelected: childName === selectedNode,
            hasChildren: ent.children && ent.children.length > 0,
          },
        });
        
        // 깊이 정보 저장
        depthNodesRef.current.set(depthNodeId, {
          nodeId: depthNodeId,
          entityName: childName,
          depth: newDepth,
          parentId: parentNodeId,
        });
        
        // 1깊이 노드는 addedEntitiesRef에도 추가 (중복 방지)
        if (newDepth === 1) {
          addedEntitiesRef.current.add(childName);
        }
        
        // 부모 → 자식 연결 (아래쪽으로)
        const edgeId = `edge-${parentNodeId}-${depthNodeId}`;
        if (!addedEdgesRef.current.has(edgeId)) {
          newEdges.push({
            id: edgeId,
            source: parentNodeId,
            sourceHandle: "bottom",
            target: depthNodeId,
            type: "default",
            style: { stroke: edgeColor, strokeWidth: 2 },
          });
          addedEdgesRef.current.add(edgeId);
        }
      });
      
      if (newNodes.length > 0) {
        setNodes((prev) => [...prev, ...newNodes]);
        setEdges((prev) => {
          const existingIds = new Set(prev.map(e => e.id));
          const uniqueNewEdges = newEdges.filter(e => !existingIds.has(e.id));
          return [...prev, ...uniqueNewEdges];
        });
        console.log(`[GraphView] 깊이 ${newDepth} 노드 ${newNodes.length}개 추가 (아래쪽)`);
      }
    } catch (err) {
      console.error(`[GraphView] 엔티티 확장 오류:`, err);
      }
  }, [selectedNode, setNodes, setEdges]);
  
  // 확장된 키워드 추적
  const expandedKeywordRef = useRef<string | null>(null);
  
  // 노드 클릭 핸들러 - 관련 사료 표시 (자동 확장은 시간 기반으로 처리)
  const handleNodeClickWithExpand = useCallback(
    async (_: any, node: Node) => {
      const displayName = node.data.label;  // 화면 표시용 이름
      const searchName = node.data.term || node.data.label;  // 검색용 실제 키워드
      const isKeyword = node.data.isKeyword;
      const nodeType = node.data.nodeType || '';
      const rootKeyword = node.data.rootKeyword;
      
      // 같은 노드 다시 클릭 → 선택 해제 (검색용 이름으로 비교)
      if (selectedNode === searchName) {
        onNodeClick(null, null);
        return;
      }
      
      // 화면 이동
      const nodeSize = isKeyword ? 50 : 30;
      focusOnPosition(node.position.x + nodeSize, node.position.y + nodeSize);
      
      // 세력 타입은 선택/하이라이트만 하고 관련사료/RAG 검색은 안됨
      // (onNodeClick에 skipRag: true 전달)
      if (nodeType === '세력') {
        console.log(`[GraphView] '${displayName}'는 '${nodeType}' 타입이므로 관련사료/질문 생략 (선택만 함)`);
        onNodeClick(searchName, {
          ...node.data,
          displayName: displayName,
          nodeType: nodeType,
          rootKeyword: rootKeyword,
          skipRag: true,  // RAG 검색 스킵 플래그
          skipSources: true,  // 관련 사료 스킵 플래그
        });
        return;
      }
      
      // 전투지역 타입은 완전히 무시 (선택도 안됨)
      if (nodeType === '전투지역') {
        console.log(`[GraphView] '${displayName}'는 '${nodeType}' 타입이므로 상호작용 없음`);
        return;
      }
      
      // 노드 클릭 시 관련 사료 표시 (onNodeClick 콜백)
      // searchName(실제 키워드)을 전달하여 검색에 사용
      onNodeClick(searchName, {
        ...node.data,
        displayName: displayName,  // 화면 표시용 이름도 전달
        nodeType: nodeType,
        rootKeyword: rootKeyword,
      });
    },
    [onNodeClick, selectedNode, focusOnPosition]
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
        
        // 5초 단위로 슬라이스 인덱스 계산 (더 정밀하게)
        const sliceDuration = 5;
        const sliceIndex = Math.floor(currentSec / sliceDuration);
        
        // 같은 슬라이스면 스킵
        if (sliceIndex === lastSliceIndexRef.current) {
          return;
        }

        console.log(`[GraphView] 슬라이스 변경: ${lastSliceIndexRef.current} -> ${sliceIndex} (currentTime: ${currentSec}초)`);
        lastSliceIndexRef.current = sliceIndex;

        // 현재 시간 기준으로 요청 (현재 시간 이전의 키워드만)
        const start = currentSec;
        const end = currentSec + sliceDuration;

        console.log(`[GraphView] API 호출: start=${start}, end=${end}`);
        const data = await fetchKeywords(videoId, start, end);

        console.log(`[GraphView] API 응답:`, data.keywords?.map((k: any) => k.term));

        if (!data.keywords || data.keywords.length === 0) {
          console.log(`[GraphView] 키워드 없음, 스킵`);
          return;
        }

        // 현재 시간에 맞는 키워드만 필터링 (start 시간 체크)
        const currentSec2 = Math.floor(currentTime);
        const timeFilteredKeywords = data.keywords.filter((k: any) => {
          // start 시간이 없거나 현재 시간이 start 이상이면 표시
          if (k.start === undefined) return true;
          return currentSec2 >= k.start;
        });

        if (timeFilteredKeywords.length === 0) {
          console.log(`[GraphView] 시간 조건 미충족, 스킵 (현재: ${currentSec2}초)`);
          return;
        }

        // 새로 추가할 키워드만 필터링
        const newKeywords = timeFilteredKeywords.filter(
          (k: any) => !addedKeywordsRef.current.has(k.term)
        );

        // 새 키워드가 없으면 스킵
        if (newKeywords.length === 0) {
          return;
        }

        console.log(`[GraphView] 새 키워드 추가:`, newKeywords.map((k: any) => k.term));

        // relations 데이터 저장 (API 응답에 있으면 - 기존에 추가)
        if (data.relations && data.relations.length > 0) {
          // 중복 방지: 이미 있는 relation은 추가하지 않음
          const existingKeys = new Set(
            relationsDataRef.current.map((r: any) => `${r.from}->${r.to}`)
          );
          const newRelations = data.relations.filter(
            (r: any) => !existingKeys.has(`${r.from}->${r.to}`)
          );
          if (newRelations.length > 0) {
            relationsDataRef.current = [...relationsDataRef.current, ...newRelations];
            console.log(`[GraphView] relations 추가:`, newRelations.map((r: any) => `${r.from}->${r.to}(${r.start}초)`));
          }
        }

        const centerX = 400; // 키워드는 화면 중앙에 배치
        const centerY = 100; // 상단에서 시작

        // 수동 children 데이터 저장 (시간 정보 포함)
        newKeywords.forEach((keyword: any) => {
          if (keyword.children && keyword.children.length > 0) {
            // children에 rootKeyword 정보 추가
            const childrenWithRoot = keyword.children.map((c: any) => ({
              ...c,
              rootKeyword: keyword.term,
            }));
            childrenDataRef.current.set(keyword.term, childrenWithRoot);
            console.log(`[GraphView] '${keyword.term}'의 children 저장:`, childrenWithRoot.map((c: any) => `${c.term}(${c.start}-${c.end})`));
            
            // 재귀적으로 하위 children도 저장 (rootKeyword 전파)
            const saveChildrenRecursive = (children: any[], rootKw: string) => {
              children.forEach((child: any) => {
                if (child.children && child.children.length > 0) {
                  const subChildrenWithRoot = child.children.map((sc: any) => ({
                    ...sc,
                    rootKeyword: rootKw,
                  }));
                  childrenDataRef.current.set(child.term, subChildrenWithRoot);
                  saveChildrenRecursive(subChildrenWithRoot, rootKw);
                }
              });
            };
            saveChildrenRecursive(childrenWithRoot, keyword.term);
          }
        });

        const newNodes: Node[] = [];
        const newEdges: Edge[] = [];

        // 메인 키워드 노드 추가 (중앙 상단에 배치)
        newKeywords.forEach((keyword: any, kwIdx: number) => {
          const keywordId = `keyword-${keyword.term}`;
          
          // 첫 번째 키워드만 표시 (메인 주제)
          if (kwIdx > 0) return;
          
          newNodes.push({
            id: keywordId,
            type: "circle",
            position: { x: centerX, y: centerY },
            sourcePosition: Position.Bottom,
            targetPosition: Position.Top,
            data: {
              label: keyword.displayName || keyword.term,  // displayName이 있으면 사용
              term: keyword.term,  // 실제 키워드 (검색용)
              score: keyword.score,
              isKeyword: true,
              isSelected: keyword.term === selectedNode,
              hasChildren: keyword.children && keyword.children.length > 0,
              nodeType: keyword.type || '사건',  // 키워드 타입 저장
              rootKeyword: null,  // 키워드 자체는 루트
            },
          });

          addedKeywordsRef.current.add(keyword.term);
        });

        console.log(`[GraphView] 생성된 노드:`, newNodes.length, `엣지:`, newEdges.length);

        // 새 노드/엣지만 추가 (기존 것 유지)
        if (newNodes.length > 0) {
          setNodes(prev => [...prev, ...newNodes]);
          setEdges(prev => [...prev, ...newEdges]);

          // 메인 키워드 노드 정중앙으로 화면 이동 (노드 크기 절반 더해서 중앙으로)
          const keywordNodeSize = 100; // 키워드 노드 크기
          setTimeout(() => {
            focusOnPosition(centerX + keywordNodeSize / 2, centerY + keywordNodeSize / 2, 0.8);
          }, 200);
        }

        // 지명 추출 및 콜백 호출 (지도 연동)
        if (onPlaceNamesExtracted) {
          // 키워드를 지도 검색 대상에 포함
          const allPlaceNames = newKeywords.map((k: any) => k.term);

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

  // 시간에 따라 자동으로 children 표시 + 동적 재배치 (전체 트리 재계산)
  useEffect(() => {
    if (!videoId || nodes.length === 0) return;
    
    const currentSec = currentTime;
    
    // 키워드 노드 찾기
    const keywordNode = nodes.find(n => n.data.isKeyword);
    if (!keywordNode) return;
    
    const rootX = keywordNode.position.x;  // 키워드 노드의 X 좌표 (중앙 기준점)
    const rootY = keywordNode.position.y;
    const rootId = keywordNode.id;
    const rootName = keywordNode.data.label;
    
    // 해당 키워드의 children 가져오기
    const depth1Children = childrenDataRef.current.get(rootName) || [];
    if (depth1Children.length === 0) return;
    
    // 현재 시간에 활성화된 1깊이 children
    const activeDepth1 = depth1Children.filter((child: any) => {
      if (child.start === undefined) return true;
      return currentSec >= child.start;
    });
    
    if (activeDepth1.length === 0) return;
    
    // 새로 추가할 노드가 있는지 확인
    const hasNewNodes = activeDepth1.some((child: any) => 
      !addedEntitiesRef.current.has(child.term)
    ) || activeDepth1.some((child: any) => {
      const d2Children = childrenDataRef.current.get(child.term) || child.children || [];
      return d2Children.some((d2: any) => {
        if (d2.start === undefined || currentSec >= d2.start) {
          return !addedEntitiesRef.current.has(d2.term);
        }
        return false;
      });
    });
    
    // 변경사항이 없으면 스킵
    if (!hasNewNodes) return;
    
    console.log(`[GraphView] 트리 재계산 - 현재 시간: ${currentSec}초`);
    
    // 깊이별 레이아웃 상수 계산 함수 (간격 축소)
    const getVerticalGap = (depth: number): number => {
      return Math.max(140 - (depth - 1) * 8, 90);
    };
    
    const getHorizontalGap = (depth: number): number => {
      return Math.max(100 - (depth - 1) * 10, 70);
    };
    
    // 전체 트리 구조 계산
    const treeNodes: { id: string; x: number; y: number; data: any; width: number }[] = [];
    const treeEdges: { id: string; source: string; target: string }[] = [];
    
    // 키워드 노드 중심 좌표
    const KEYWORD_SIZE = getNodeSizeByDepth(0, true);
    
    // 트리 구조 정보 (너비 계산용)
    interface TreeNodeInfo {
      id: string;
      name: string;
      depth: number;
      parentId: string;
      children: TreeNodeInfo[];
      width: number;  // 자식 포함 전체 너비
      x: number;      // 중심 X 좌표
      y: number;
      data: any;
    }
    
    // 1단계: 트리 구조 빌드 (재귀)
    const buildTreeStructure = (
      parentId: string,
      children: any[],
      depth: number,
      maxDepth: number = 10
    ): TreeNodeInfo[] => {
      if (depth > maxDepth || children.length === 0) return [];
      
      // 현재 시간에 활성화된 children 필터링
      // 한번 추가된 노드(addedEntitiesRef에 있는 노드)는 항상 포함
      const activeChildren = children.filter((child: any) => {
        // 이미 추가된 노드는 항상 표시
        if (addedEntitiesRef.current.has(child.term)) return true;
        // 시작 시간이 없으면 항상 표시
        if (child.start === undefined) return true;
        // 현재 시간이 시작 시간 이상이면 표시
        return currentSec >= child.start;
      });
      
      if (activeChildren.length === 0) return [];
      
      // 지명/전투지역을 맨 왼쪽으로 정렬
      const sortedChildren = [...activeChildren].sort((a: any, b: any) => {
        const aType = (a.type || '').toLowerCase();
        const bType = (b.type || '').toLowerCase();
        const aIsPlace = aType === '지명' || aType === '전투지역';
        const bIsPlace = bType === '지명' || bType === '전투지역';
        
        if (aIsPlace && !bIsPlace) return -1;
        if (!aIsPlace && bIsPlace) return 1;
        return 0;
      });
      
      const nodeSize = getNodeSizeByDepth(depth);
      const horizontalGap = getHorizontalGap(depth);
      
      return sortedChildren.map((child: any) => {
        const childName = child.term;
        const nodeId = `depth${depth}-${childName}`;
        
        // children 데이터 저장
        if (child.children && child.children.length > 0) {
          childrenDataRef.current.set(childName, child.children);
        }
        
        const subChildren = childrenDataRef.current.get(childName) || [];
        const nodeType = child.type || '미분류';
        const categoryType = getCategoryType('', nodeType);
        
        // 재귀적으로 하위 트리 빌드
        const childNodes = buildTreeStructure(nodeId, subChildren, depth + 1, maxDepth);
        
        // 너비 계산: 자식이 있으면 자식들의 총 너비, 없으면 자신의 크기
        let width = nodeSize;
        if (childNodes.length > 0) {
          const childrenTotalWidth = childNodes.reduce((sum, c) => sum + c.width, 0);
          const gaps = (childNodes.length - 1) * getHorizontalGap(depth + 1);
          width = Math.max(nodeSize, childrenTotalWidth + gaps);
        }
        
        return {
          id: nodeId,
          name: childName,
          depth: depth,
          parentId: parentId,
          children: childNodes,
          width: width,
          x: 0,  // 나중에 계산
          y: 0,  // 나중에 계산
          data: {
            label: child.displayName || childName,  // displayName이 있으면 사용
            term: childName,  // 실제 키워드 (검색용)
            isKeyword: false,
            depth: depth,
            parentId: parentId,
            isSelected: childName === selectedNode,
            hasChildren: subChildren.length > 0,
            nodeType: nodeType,
            rootKeyword: child.rootKeyword || rootName,
            categoryType: categoryType,
          },
        };
      });
    };
    
    // 2단계: 위치 계산 (루트 중심 기준으로 배치)
    const calculatePositions = (
      nodes: TreeNodeInfo[],
      parentCenterX: number,
      parentY: number,
      depth: number
    ) => {
      if (nodes.length === 0) return;
      
      const verticalGap = getVerticalGap(depth);
      const horizontalGap = getHorizontalGap(depth);
      const nodeSize = getNodeSizeByDepth(depth);
      const childY = parentY + verticalGap;
      
      // 전체 너비 계산 (각 노드의 width 합 + 간격)
      const totalWidth = nodes.reduce((sum, n) => sum + n.width, 0) + (nodes.length - 1) * horizontalGap;
      
      // 시작 X 좌표 (부모 중심 기준 왼쪽으로)
      let currentX = parentCenterX - totalWidth / 2;
      
      nodes.forEach((node) => {
        // 노드 중심 X 좌표 (자신의 너비 중앙)
        const nodeCenterX = currentX + node.width / 2;
        node.x = nodeCenterX - nodeSize / 2;  // 좌상단 좌표
        node.y = childY;
        
        // 자식 노드 위치 계산 (재귀)
        if (node.children.length > 0) {
          calculatePositions(node.children, nodeCenterX, childY, depth + 1);
        }
        
        // 다음 노드 시작 위치
        currentX += node.width + horizontalGap;
      });
    };
    
    // 3단계: 트리 노드를 flat 배열로 변환
    const flattenTree = (nodes: TreeNodeInfo[]) => {
      nodes.forEach((node) => {
        treeNodes.push({
          id: node.id,
          x: node.x,
          y: node.y,
          data: node.data,
          width: node.width,
        });
        
        treeEdges.push({
          id: `edge-${node.parentId}-${node.id}`,
          source: node.parentId,
          target: node.id,
        });
        
        if (node.children.length > 0) {
          flattenTree(node.children);
        }
      });
    };
    
    // 트리 빌드 및 위치 계산
    const treeStructure = buildTreeStructure(rootId, activeDepth1, 1);
    const rootCenterX = rootX + KEYWORD_SIZE / 2;
    calculatePositions(treeStructure, rootCenterX, rootY, 1);
    flattenTree(treeStructure);
    
    // 새로 추가될 노드 미리 계산 (화면 이동용)
    const newNodesList: { x: number; y: number; label: string; depth: number }[] = [];
    
    // 기존에 추가된 노드 확인
    const existingNodeIds = new Set(nodes.map(n => n.id));
    
    treeNodes.forEach(treeNode => {
      if (!existingNodeIds.has(treeNode.id) && !addedEntitiesRef.current.has(treeNode.data.label)) {
        newNodesList.push({
          x: treeNode.x,
          y: treeNode.y,
          label: treeNode.data.label,
          depth: treeNode.data.depth,
        });
      }
    });
    
    // 노드 업데이트
    setNodes(prev => {
      let result = [...prev];
      
      treeNodes.forEach(treeNode => {
        const existingIdx = result.findIndex(n => n.id === treeNode.id);
        
        if (existingIdx !== -1) {
          // 기존 노드 위치 업데이트
          result[existingIdx] = {
            ...result[existingIdx],
            position: { x: treeNode.x, y: treeNode.y },
            data: { ...result[existingIdx].data, ...treeNode.data },
          };
        } else {
          // 새 노드 추가
          result.push({
            id: treeNode.id,
            type: "circle",
            position: { x: treeNode.x, y: treeNode.y },
            data: treeNode.data,
          });
          
          // 추적 정보 저장
          depthNodesRef.current.set(treeNode.id, {
            nodeId: treeNode.id,
            entityName: treeNode.data.label,
            depth: treeNode.data.depth,
            parentId: treeNode.data.parentId,
          });
          addedEntitiesRef.current.add(treeNode.data.label);
          
          console.log(`[GraphView] 노드 추가: ${treeNode.data.label} (깊이 ${treeNode.data.depth})`);
        }
      });
      
      return result;
    });
    
    // 엣지 업데이트
    console.log(`[GraphView] 엣지 추가 시도: ${treeEdges.length}개`, treeEdges.map(e => `${e.source} → ${e.target}`));
    
    setEdges(prev => {
      let result = [...prev];
      
      treeEdges.forEach(treeEdge => {
        // 이미 추가된 엣지인지 확인
        const alreadyExists = result.some(e => e.id === treeEdge.id);
        if (!alreadyExists) {
          result.push({
            id: treeEdge.id,
            source: treeEdge.source,
            sourceHandle: "bottom",
            target: treeEdge.target,
            type: "default",
            style: { stroke: COLORS.edge, strokeWidth: 2 },
          });
          console.log(`[GraphView] 엣지 추가: ${treeEdge.source} → ${treeEdge.target}`);
        }
      });
      
      return result;
    });
    
    // 새 노드가 추가되면 마지막 새 노드로 화면 이동
    if (newNodesList.length > 0) {
      const lastNewNode = newNodesList[newNodesList.length - 1];
      // 깊이에 따른 노드 크기
      const nodeSize = getNodeSizeByDepth(lastNewNode.depth);
          setTimeout(() => {
        console.log(`[GraphView] 화면 이동: ${lastNewNode.label} (${lastNewNode.x + nodeSize/2}, ${lastNewNode.y + nodeSize/2})`);
        focusOnPosition(lastNewNode.x + nodeSize / 2, lastNewNode.y + nodeSize / 2, 0.75);
      }, 200);
    }
    
  }, [currentTime, videoId, nodes, selectedNode, setNodes, setEdges, focusOnPosition]);

  // 시간에 따라 relations 처리 (두 키워드 연결)
  useEffect(() => {
    if (!videoId || relationsDataRef.current.length === 0) return;
    
    const currentSec = currentTime;
    const relations = relationsDataRef.current;
    
    // 현재 시간에 활성화되어야 할 relations 확인
    relations.forEach((relation: any) => {
      const relationId = `relation-${relation.from}-${relation.to}`;
      
      // 이미 추가된 relation이면 스킵
      if (addedRelationsRef.current.has(relationId)) return;
      
      // 시간 조건 확인
      if (relation.start !== undefined && currentSec < relation.start) return;
      
      // 두 노드가 모두 존재하는지 확인 (term 또는 label로 검색)
      const fromNode = nodes.find(n => 
        n.data.term === relation.from || 
        n.data.label === relation.from
      );
      const toNode = nodes.find(n => 
        n.data.term === relation.to || 
        n.data.label === relation.to
      );
      
      if (!fromNode || !toNode) {
        // 디버그: 어떤 노드가 없는지 확인 (5초마다만 로그)
        if (Math.floor(currentSec) % 5 === 0) {
          const nodeTerms = nodes.map(n => n.data.term || n.data.label);
          console.log(`[GraphView] relation 대기 (${currentSec}초): ${relation.from} -> ${relation.to}`);
          console.log(`[GraphView] 현재 노드들:`, nodeTerms);
          console.log(`[GraphView] fromNode(${relation.from}): ${fromNode ? '있음' : '없음'}, toNode(${relation.to}): ${toNode ? '있음' : '없음'}`);
        }
        return;
      }
      
      console.log(`[GraphView] relation 추가: ${relation.from} -> ${relation.to} (${relation.label || '연결'})`);
      
      // relation 엣지 추가 (점선, 다른 색상으로 구분)
      const edgeId = `relation-edge-${fromNode.id}-${toNode.id}`;
      
      setEdges(prev => {
        // 이미 존재하면 스킵
        if (prev.some(e => e.id === edgeId)) return prev;
        
        return [...prev, {
          id: edgeId,
          source: fromNode.id,
          target: toNode.id,
          type: "default",
          label: relation.label || '',
          labelStyle: { fill: '#fbbf24', fontWeight: 600, fontSize: 12 },
          labelBgStyle: { fill: '#131416', fillOpacity: 1 },
          labelBgPadding: [4, 8] as [number, number],
          style: { 
            stroke: '#fbbf24',  // 노란색 (relation 전용)
            strokeWidth: 2,
            strokeDasharray: '5,5',  // 점선
          },
          animated: true,  // 애니메이션 효과
        }];
      });
      
      // 추가된 relation 기록
      addedRelationsRef.current.add(relationId);
    });
    
  }, [currentTime, videoId, nodes, setEdges]);

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
    expandedKeywordRef.current = null;
    expanded2DepthRef.current = null;
    childrenDataRef.current.clear();  // 수동 children 데이터도 초기화
    relationsDataRef.current = [];  // relations 데이터도 초기화
    addedRelationsRef.current.clear();  // 추가된 relations도 초기화
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
      
      {/* 색상 범례 (오른쪽 하단) */}
      {nodes.length > 0 && (
        <div
          style={{
            position: "absolute",
            bottom: 12,
            right: 12,
            padding: "8px 12px",
            background: "rgba(30, 41, 59, 0.95)",
            borderRadius: "8px",
            zIndex: 10,
            fontSize: "10px",
            display: "flex",
            flexDirection: "column",
            gap: "4px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.keyword }} />
            <span style={{ color: "#e2e8f0" }}>주제</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.place }} />
            <span style={{ color: "#e2e8f0" }}>지명</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.faction }} />
            <span style={{ color: "#e2e8f0" }}>세력</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.person }} />
            <span style={{ color: "#e2e8f0" }}>인물</span>
          </div>
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
