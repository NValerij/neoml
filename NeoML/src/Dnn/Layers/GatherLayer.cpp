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
    CPtr<CDnnBlob> weights = inputBlobs[0];
    CPtr<const CDnnBlob> indexes = inputBlobs[1];
    const CPtr<CDnnBlob>& result = outputBlobs[0];

    // Replace weights and indexes with padded version if paddings are enabled.
    addOptionalPaddings( weights, indexes );

    // Shifting indexes to make it flat
    CFloatHandleStackVar shiftedIndexes( MathEngine(), indexes->GetDataSize() );
    flatternIndexes( *indexes, shiftedIndexes );

    // Copying data to output table
    CLookupDimension dims{ weights->GetObjectCount(), weights->GetObjectSize() };
    MathEngine().VectorMultichannelLookupAndCopy(
        indexes->GetDataSize(), 1, shiftedIndexes,
        &weights->GetData<const float>(), &dims, 1, result->GetData(), result->GetObjectSize() );
}

void CGatherLayer::BackwardOnce()
{
    CPtr<CDnnBlob> weightsDiff = inputDiffBlobs[0];
    CPtr<const CDnnBlob> indexes = inputBlobs[1];

    // Replace weights and indexes with padded version if paddings are enabled.
    addOptionalPaddings( weightsDiff, indexes );

    const CDnnBlob* resultDiff = outputDiffBlobs[0];
    NeoAssert( resultDiff != nullptr );

    // Shifting indexes to make it flat
    CFloatHandleStackVar shiftedIndexes( MathEngine(), indexes->GetDataSize() );
    flatternIndexes( *indexes, shiftedIndexes );

    // Gathering gradients for used vectors
    CLookupDimension dims{ weightsDiff->GetObjectCount(), weightsDiff->GetObjectSize() };
    CFloatHandleStackVar learningRate( MathEngine() );
    learningRate.SetValue( CBaseLayer::GetBaseLearningRate() );
    MathEngine().VectorMultichannelLookupAndAddToTable(
        indexes->GetDataSize(), 1, shiftedIndexes,
        &weightsDiff->GetData(), &dims, 1, learningRate,
        resultDiff->GetData(), resultDiff->GetObjectSize() );

    if( arePaddingsUsed ) {
        // Removing paddings diff from weightsDiff
        MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(), weightsDiff->GetObjectData( 1 ), inputDiffBlobs[0]->GetDataSize() );
    }
}

// Adding padding vector to weights and shifting indexes by one if paddings are enabled.
void CGatherLayer::addOptionalPaddings( CPtr<CDnnBlob>& weights, CPtr<const CDnnBlob>& indexes ) const
{
    if( !arePaddingsUsed ) {
        return;
    }

    // Adding a zero padding vector at first place
    CBlobDesc desc = weights->GetDesc();
    desc.SetDimSize( BD_BatchLength, 1 );
    desc.SetDimSize( BD_BatchWidth, 1 + weights->GetObjectCount() );
    CPtr<CDnnBlob> weightsEx = CDnnBlob::CreateBlob( MathEngine(), desc );
    // Zero padding vector
    MathEngine().VectorFill( weightsEx->GetData(), 0.f, weights->GetObjectSize() );
    // Other weights
    MathEngine().VectorCopy( weightsEx->GetObjectData( 1 ), weights->GetData(), weights->GetDataSize() );
    weights = weightsEx;

    // Shifting indexes by one
    CPtr<CDnnBlob> shiftedIndexes = indexes->GetClone();
    CFloatHandleStackVar one( MathEngine() );
    one.SetValue( 1.f );
    MathEngine().VectorAddValue( indexes->GetData(), shiftedIndexes->GetData(), indexes->GetDataSize(), one );
    indexes = shiftedIndexes;
}

// Converting in-sample indexes to global in batch for lookup operation
// (like lookup in 2-D embeddings table).
void CGatherLayer::flatternIndexes( const CDnnBlob& indexes, CFloatHandleStackVar& result ) const
{
    NeoAssert( result.Size() == indexes.GetDataSize() );

    // Every sample index is shifted by total samples count because they are group the way
    // then first (second, etc.) elements of every sample goes alltogether.
    CFloatHandleStackVar shifts( MathEngine() );
    shifts.SetValue( static_cast<float>( indexes.GetBatchWidth() ) ); // by total samples in batch (same for weights and indexes)
    MathEngine().VectorMultiply( indexes.GetData(), result, result.Size(), shifts );

    // Tricky operation which adds its column number to every element.
    // Allow us to distinguish different samples in a single row.
    CBlobDesc desc = indexes.GetDesc();
    desc.SetDimSize( BD_Width, indexes.GetBatchWidth() );
    desc.SetDimSize( BD_BatchWidth, 1 );
    MathEngine().AddWidthIndex( desc, result, true, result );
}

CLayerWrapper<CGatherLayer> Gather()
{
	return CLayerWrapper<CGatherLayer>( "Gather" );
}

} // namespace NeoML
