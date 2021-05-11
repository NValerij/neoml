/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/GatherLayer.h>

namespace NeoML {

static const float GELUMultiplier = 1.702f;

CGatherLayer::CGatherLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CGatherLayer", false )
{
}

static const int CGatherLayerVersion = 0;

void CGatherLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( CGatherLayerVersion );
    (void) version; // not used yet;

	CBaseLayer::Serialize( archive );
    archive.Serialize( arePaddingsUsed );
}

void CGatherLayer::Reshape()
{
	CheckInputs();

    CheckArchitecture( inputDescs.Size() == 2, GetName(), "Bad inputs count" );
	const CBlobDesc& weightsDesc = inputDescs[0];
	const CBlobDesc& indexesDesc = inputDescs[1];

    CheckArchitecture( weightsDesc.GetDataType() == CT_Float, GetName(), "Bad weights blob type" );
    CheckArchitecture( weightsDesc.GetDataType() == CT_Float, GetName(), "Bad indexes blob type" );
    CheckArchitecture( weightsDesc.BatchWidth() == indexesDesc.BatchWidth(), GetName(),
        "Weights samples count differs from index samples count" );
    CheckArchitecture( weightsDesc.ListSize() == 1, GetName(), "Bad weights ListSize" );
    CheckArchitecture( indexesDesc.ListSize() == 1, GetName(), "Bad indexes ListSize" );
    CheckArchitecture( indexesDesc.ObjectSize() == 1, GetName(), "Bad indexes object size" );

	outputDescs.SetSize( 1 );
	outputDescs[0] = weightsDesc;
    outputDescs[0].SetDimSize( BD_BatchLength, indexesDesc.BatchLength() );
}

void CGatherLayer::RunOnce()
{
    CPtr<const CDnnBlob> weights = inputBlobs[0];
    CPtr<const CDnnBlob> indexes = inputBlobs[1];
    CPtr<CDnnBlob> result = outputBlobs[0];

    // if( arePaddingsUsed ) {
    //     static_assert( false, "Add padding vector and shift indexes by one" );
    // }

    // Shifting indexes to make it flat
    CFloatHandleStackVar shiftedIndexes( MathEngine(), indexes->GetDataSize() );
    flatternIndexes( *indexes, weights->GetBatchLength(), shiftedIndexes );

    // Copying data to output table
    CLookupDimension dims{ weights->GetObjectCount(), weights->GetObjectSize() };
    MathEngine().VectorMultichannelLookupAndCopy(
        indexes->GetDataSize(), 1, shiftedIndexes,
        &weights->GetData(), &dims, 1, result->GetData(), result->GetObjectSize() );
}

void CGatherLayer::BackwardOnce()
{
    CPtr<CDnnBlob> weightsDiff = inputDiffBlobs[0];
    CPtr<CDnnBlob> indexes = inputBlobs[1];

    // if( arePaddingsUsed ) {
    //     static_assert( false, "Shift indexes by one and add place for padding diff" );
    // }

    const CDnnBlob* resultDiff = outputDiffBlobs[0];
    NeoAssert( resultDiff != nullptr );

    // Shifting indexes to make it flat
    CFloatHandleStackVar shiftedIndexes( MathEngine(), indexes->GetDataSize() );
    flatternIndexes( *indexes, weightsDiff->GetBatchLength(), shiftedIndexes );

    // Gathering gradients for used vectors
    CLookupDimension dims{ weightsDiff->GetObjectCount(), weightsDiff->GetObjectSize() };
    CFloatHandleStackVar learningRate( MathEngine() );
    learningRate.SetValue( CBaseLayer::GetBaseLearningRate() );
    MathEngine().VectorMultichannelLookupAndAddToTable(
        indexes->GetDataSize(), 1, shiftedIndexes,
        &weightsDiff->GetData(), &dims, 1, learningRate,
        resultDiff->GetData(), resultDiff->GetObjectSize() );

    // if( arePaddingsUsed ) {
    //     static_assert( false, "Remove padding diff from weightsDiff" );
    // }
}

// Converting in-sample indexes to global in batch for lookup operation
// (like lookup in 2-D embeddings table).
void CGatherLayer::flatternIndexes( const CDnnBlob& indexes, int weightsBatchLength, CFloatHandleStackVar& result ) const
{
    NeoAssert( result.Size() == indexes.GetDataSize() );

    // i-th row contains from i-th item from every sample.
    // every j-th sample has indexes shifted by j * batch_length (sample length)
    CFloatHandleStackVar inRowShifts( MathEngine(), indexes.GetBatchLength() );
    CArray<float> shiftValues;
    shiftValues.Add( static_cast<float>( weightsBatchLength ), indexes.GetBatchLength() );
    shiftValues[0] = 0.f;
    for (int i = 1; i < shiftValues.Size(); i++)
    {
        shiftValues[i] += shiftValues[i - 1];
    }
    MathEngine().DataExchangeTyped( inRowShifts.GetHandle(), shiftValues.GetPtr(), shiftValues.Size() );

    MathEngine().AddVectorToMatrixRows( 1, indexes.GetData(),
        result, indexes.GetBatchLength(), indexes.GetBatchWidth(), inRowShifts );
}

CLayerWrapper<CGatherLayer> Gather()
{
	return CLayerWrapper<CGatherLayer>( "Gather" );
}

} // namespace NeoML
