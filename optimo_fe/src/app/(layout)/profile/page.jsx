"use client";

import React, { useEffect, useState } from "react";
import Image from 'next/image';
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import {
    useReactTable,
    getCoreRowModel,
    flexRender,
    createColumnHelper,
} from "@tanstack/react-table";
import { useAuth } from "@/contexts/AuthContext";

import { api } from "@/lib/api";
import { api as chatapi } from "@/lib/chatapi";

const data = [
    { model: "GPT-4.1", requests: 2524, powerWh: "100.96Wh", perMessageWh: "0.04Wh" },
    { model: "GPT-4.1 mini", requests: 1345, powerWh: "67.25Wh", perMessageWh: "0.05Wh" },
    { model: "GPT-4.1 nano", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
    { model: "o4-mini (high)", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
    { model: "GPT-4.5", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
    { model: "GPT-4.5", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
    { model: "GPT-4.5", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
    { model: "GPT-4.5", requests: 234, powerWh: "14.04Wh", perMessageWh: "0.06Wh" },
];

const columnHelper = createColumnHelper();
const columns = [
    columnHelper.accessor("model", { header: "AI 모델" }),
    columnHelper.accessor("requests", { header: "요청 수" }),
    columnHelper.accessor("powerWh", { header: "전력 소모" }),
    columnHelper.accessor("perMessageWh", { header: "Wh per message" }),
];

export default function ProfilePage() {
    const { user, isLoading } = useAuth();
    const [userInfo, setUserInfo] = useState(null);
    const [electricityData, setElectricityData] = useState([]);

    useEffect(() => {
        const fetchUserInfo = async () => {
            try {
                const info = await api.get('/user/jwt');
                console.log("userInfo", info);
                setUserInfo(info);
            } catch (error) {
                console.error("사용자 정보 가져오기 실패:", error);
            }
        };

        fetchUserInfo();
    }, []);

    useEffect(() => {
        const fetchData = async () => {
            const response = await chatapi.get('/electricity');
            setElectricityData(response);
        }
        fetchData();
    }, []);

    const table = useReactTable({
        data,
        columns,
        getCoreRowModel: getCoreRowModel(),
    });

    const totalRequests = data.reduce((acc, row) => acc + row.requests, 0);
    const totalPowerWh = data.reduce((acc, row) => {
        const val = parseFloat(row.powerWh.replace("Wh", ""));
        return acc + (isNaN(val) ? 0 : val);
    }, 0).toFixed(2);

    // 로딩 중이거나 사용자가 없으면 로딩 화면 표시
    if (isLoading || !user) {
        console.log("isLoading",isLoading);
        console.log("user",user);
        return (
            <div className="flex items-center justify-center h-screen text-white">
                <div>로딩 중...</div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-screen text-white">
            <div className="text-white w-full flex flex-col items-center pb-16">
                <div className="w-full h-64 relative">
                    <Image
                        src="/icon/background.png"
                        alt="background"
                        fill
                        className="object-cover"
                    />
                </div>
                <div className="-mt-20 z-10">
                    <Image
                        src={'/userProfile/' + (parseInt(user.profileImage)+1) + '.png' || `/userProfile/1.png`}
                        alt="profile"
                        width={150}
                        height={150}
                        className="rounded-full"
                    />
                </div>
                <div className="mt-4 text-center">
                    <p className="text-xl font-bold">{user.name || "No name"}</p>
                    <p className="text-gray-400 underline">{user.email}</p>
                    <p className="text-sm text-gray-500 mt-1">
                        총 전력 소비: {electricityData?.totalUsedEstimatedElectricity?.toFixed(2)}Wh | 가장 큰 LLM 모델 선택시의 전력 소비: {electricityData?.totalLLMEstimatedElectricity?.toFixed(2)}Wh
                    </p>
                </div>

            </div>
            <div className="flex-1 flex items-center justify-center">
                <div className="space-y-8 p-8 flex-1 flex flex-col items-center justify-center">
                    <h3 className="text-2xl font-bold">지속 가능성 대시보드</h3>
                    <p className="text-sm text-gray-400">
                        가장 많이 쓰이는 LLM, GPT-4보다 약 {electricityData?.totalSavedEstimatedElectricity?.toFixed(2)}Wh를 절약했어요 ❗️
                    </p>

                    {/* 게이지 그래프 */}
                    <div className="flex gap-10">
                        <GaugeCard title="전력 소비량" percentage={(electricityData?.totalUsedEstimatedElectricity/electricityData?.totalLLMEstimatedElectricity*100).toFixed(2)} />
                        <div className="flex flex-col justify-center items-center min-w-80 bg-neutral-800 p-4 rounded-xl shadow-lg">
                            <p className="text-2xl">CO₂ 절감량</p>
                            <p className="text-[50px] font-bold">{(electricityData?.totalSavedEstimatedElectricity*0.475).toFixed(3)}g</p>
                        </div>
                    </div>

                    {/* <div className="bg-neutral-800 p-6 rounded-xl shadow-lg max-w-3xl mx-auto w-full flex flex-col">
                        <h2 className="text-xl font-semibold mb-4 text-white">AI 모델별 요청 및 전력 소모</h2>
                        <div className="overflow-x-auto rounded-xl border border-gray-600 flex flex-col">
                            <table className="min-w-[720px] table-auto text-white">
                                <thead className="bg-neutral-700">
                                    <tr>
                                        <th className="p-3 text-left border-b border-gray-600">AI 모델</th>
                                        <th className="p-3 text-left border-b border-gray-600">요청 수</th>
                                        <th className="p-3 text-left border-b border-gray-600">전력 소모</th>
                                    </tr>
                                </thead>
                            </table>
                            <div className="max-h-[240px] overflow-y-auto">
                                <table className="min-w-[720px] table-auto text-white">
                                    <tbody>
                                        {data.map((row, idx) => (
                                            <tr key={idx} className="hover:bg-neutral-700 transition">
                                                <td className="p-3 border-b border-gray-700">{row.model}</td>
                                                <td className="p-3 border-b border-gray-700">{row.requests.toLocaleString()}</td>
                                                <td className="p-3 border-b border-gray-700">{row.powerWh} / {row.perMessageWh} per message</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <table className="min-w-[720px] table-auto text-white bg-neutral-700 rounded-b-xl border-t border-gray-600">
                                <tfoot>
                                    <tr>
                                        <td className="p-3 font-semibold">Total</td>
                                        <td className="p-3 font-semibold">{totalRequests.toLocaleString()}</td>
                                        <td className="p-3 font-semibold">{totalPowerWh}Wh</td>
                                    </tr>
                                </tfoot>
                            </table>
                        </div>
                    </div> */}
                </div>
            </div>
        </div>
    );
}

function GaugeCard({ title, percentage }) {
    return (
        <div className="flex flex-row p-16 gap-16 bg-neutral-800 rounded-xl items-center justify-center shadow-lg">
            <div className="w-24 h-24">
                <CircularProgressbar
                    value={percentage}
                    text={`${percentage}%`}
                    styles={buildStyles({
                        textColor: "white",
                        pathColor: "#A3E635",
                        trailColor: "#444",
                    })}
                />
            </div>
            <div className="flex flex-col gap-2 mt-4 text-sm text-center leading-tight">
                <div className="font-bold text-lg">{title}</div>
                <div className="text-sm text-gray-400">기준치 대비<br />현재 예상량</div>
            </div>
        </div>
    );
}
